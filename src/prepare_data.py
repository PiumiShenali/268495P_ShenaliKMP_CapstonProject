"""Step 2: data audit and leakage-safe train/validation split.

Two ways to run it, from the project root, after `python src/download_data.py`:
  - all at once:      python src/prepare_data.py
  - cell by cell:     open this file in VS Code and click "Run Cell" above each
                      "# %%" line, top to bottom (needs the Jupyter extension)

Each part prints what the project notes expect, so you can compare line by line.
The phrases are never changed in the saved files: no lowercasing, stopword
removal or stemming, because "into" vs "at" or "bomber" vs "bombers" can be
exactly what decides the label.
"""

# %% Setup: imports and settings
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (LABEL2ID, LABELS, PROCESSED_DIR, RAW_DIR, REFERENCE_COLUMNS,
                    REVERSE_LABEL, SPLITS_DIR, group_key, normalise, read_reference)

SEED = 42
N_SPLITS = 7  # validation = one fold out of seven, about 14% of train
SPLIT_FILE = SPLITS_DIR / "train_val_split.csv"


def counts(labels):
    """Label counts as text, always in the fixed label order."""
    n = labels.value_counts()
    return ", ".join(f"{label} {int(n.get(label, 0))}" for label in LABELS)


# %% Part 1: load the reference files
print("=== Part 1: load the reference files ===")
train = read_reference(RAW_DIR / "dico_nli_train_track1_reference.csv")
dev = read_reference(RAW_DIR / "dico_nli_dev_track1_reference.csv")
for name, df in [("train", train), ("dev", dev)]:
    assert list(df.columns) == REFERENCE_COLUMNS, f"{name}: unexpected columns {list(df.columns)}"
    print(f"{name}: {len(df):,} rows, {df.shape[1]} columns")
print("columns:", ", ".join(REFERENCE_COLUMNS))
print("notes expect: train 3,042 rows, dev 660 rows, 8 columns")
train.head()  # shown as a table when run cell by cell; ignored when run as a script


# %% Part 2: basic quality checks
print("\n=== Part 2: basic quality checks ===")
for name, df in [("train", train), ("dev", dev)]:
    missing = {col: int(n) for col, n in df.isna().sum().items() if n > 0}
    no_partner = df.reverse_pair_id.isna()
    print(f"{name}: missing values: {missing or 'none'}")
    print(f"{name}: rows without a reverse partner: {int(no_partner.sum())}, "
          f"all of them NEGATIVE_OTHER: {bool((df.label[no_partner] == 'NEGATIVE_OTHER').all())}, "
          f"NEGATIVE_OTHER rows in total: {int((df.label == 'NEGATIVE_OTHER').sum())}")

    texts = pd.concat([df.text1, df.text2])
    print(f"{name}: empty phrases: {int((texts.str.strip() == '').sum())}, "
          f"leading/trailing spaces: {int((texts != texts.str.strip()).sum())}, "
          f"double spaces: {int(texts.str.contains('  ', regex=False).sum())}")
    print(f"{name}: languages: {sorted(set(df.text1_lang) | set(df.text2_lang))}, "
          f"unknown labels: {int((~df.label.isin(LABELS)).sum())}, "
          f"duplicate instance_ids: {int(df.instance_id.duplicated().sum())}")

    # Each reversible row's partner must point back to it, have the two
    # phrases swapped, and carry the reversed label (FORWARD <-> BACKWARD).
    linked = df[df.reverse_pair_id.notna()]
    pairs = linked.merge(df, left_on="reverse_pair_id", right_on="instance_id", suffixes=("", "_rev"))
    correct = ((pairs.reverse_pair_id_rev == pairs.instance_id)
               & (pairs.text1_rev == pairs.text2)
               & (pairs.text2_rev == pairs.text1)
               & (pairs.label_rev == pairs.label.map(REVERSE_LABEL)))
    print(f"{name}: reversible rows whose partner is correct: {int(correct.sum())} of {len(linked)}")
print("notes expect: missing values only in reverse_pair_id (train 478, dev 106), exactly the "
      "NEGATIVE_OTHER rows; no empty phrases or extra spaces, so no text cleaning is needed")


# %% Part 3: group key (same phrase pair, ignoring order, capitals and punctuation)
print("\n=== Part 3: group key ===")
for df in (train, dev):
    df["group"] = [group_key(a, b) for a, b in zip(df.text1, df.text2)]
first = train.iloc[0]
print(f'example: "{first.text1}" + "{first.text2}"  ->  group "{first.group}"')
print(f"train: {train.group.nunique():,} groups vs {train.pair_id.nunique():,} pair_ids")
print("expected: 1,724 groups vs 1,760 pair_ids, fewer groups because copies of a pair have "
      "different pair_ids. The notes say 1,725 groups: they ignore capitals but not punctuation, "
      'so they keep "for gang-rape" and "for gang rape" apart')

# Ignoring punctuation must only merge true copies. Print every merge that
# ignoring capitals alone would not make, so a person can check each one.
both = pd.concat([train.assign(file="train"), dev.assign(file="dev")], ignore_index=True)
both["capitals_only_key"] = [" || ".join(sorted([a.lower().strip(), b.lower().strip()]))
                             for a, b in zip(both.text1, both.text2)]
punctuation_merges = both[both.groupby("group").capitals_only_key.transform("nunique") > 1]
print(f"groups that exist only because punctuation is ignored: {punctuation_merges.group.nunique()}")
print(punctuation_merges.sort_values(["group", "file"])[["file", "text1", "text2", "label"]]
      .to_string(index=False))
print('expected: 2, both true copies: "for gang-rape" / "for gang rape" (train), and '
      '"A close-up view" / "A close up view" (dev and train)')


# %% Part 4: duplicates and label conflicts
print("\n=== Part 4: duplicates and label conflicts ===")
# A copy is the same phrase pair in the same direction. A pair (A, B) and its
# reverse (B, A) are not copies: both directions are meant to be in the data.
exact = train.text1 + " -> " + train.text2
as_compared = train.text1.map(normalise) + " -> " + train.text2.map(normalise)
for name, rows in [("exact", exact), ("ignoring capitals and punctuation", as_compared)]:
    print(f"train, {name}: {int(rows.duplicated().sum())} extra copies of a phrase pair "
          f"({int(rows.duplicated(keep=False).sum())} rows involved)")
print("expected: exact 51 extra copies (94 rows), as in the notes. Ignoring capitals and "
      'punctuation, as the group key does ("A dog" / "a dog", "gang-rape" / "gang rape"): '
      "68 (126 rows). All copies are kept, because the group key puts every copy on the same "
      "side of the split")


def labels_in_key_order(df):
    """Each row's label, rewritten for the direction of its group key.

    Every label is read as if the pair were written in its key's (alphabetical)
    order, so (A, B) = FORWARD and its reverse (B, A) = BACKWARD give the same answer.
    """
    in_key_order = pd.Series([normalise(a) <= normalise(b) for a, b in zip(df.text1, df.text2)],
                             index=df.index)
    return df.label.where(in_key_order, df.label.map(REVERSE_LABEL))


# A group that still has more than one answer has contradictory labels.
answers_per_group = labels_in_key_order(train).groupby(train.group).nunique()
conflict_groups = set(answers_per_group[answers_per_group > 1].index)

conflict_rows = train[train.group.isin(conflict_groups)]
print(f"train: {len(conflict_groups)} group(s) with conflicting labels, {len(conflict_rows)} rows:")
print(conflict_rows[["text1", "text2", "label"]].to_string(index=False))
print('notes expect: 1 conflict, "South Africans" -> "South Africa" labelled both '
      "FORWARD_ENTAILMENT and NEGATIVE_OTHER")


# %% Part 5: remove the conflicting group(s)
print("\n=== Part 5: remove the conflicting group(s) ===")
is_conflict = train.group.isin(conflict_groups)
removed = train[is_conflict]
train = train[~is_conflict].reset_index(drop=True)
print(f"removed {len(removed)} rows, {len(train):,} left")
print("notes expect: 3 rows removed, 3,039 left (two different 'correct' answers "
      "would teach the model a contradiction)")


# %% Part 6: flag dev rows whose phrase pair is also in train
print("\n=== Part 6: flag dev rows whose phrase pair is also in train ===")
dev["in_train"] = dev.group.isin(set(train.group))
clean = dev[~dev.in_train]
print(f"dev rows also in train: {int(dev.in_train.sum())} "
      f"({dev.loc[dev.in_train, 'group'].nunique()} phrase pairs); clean test view: {len(clean)} rows")
# A pair and its reverse share a group, so they are flagged together. The
# clean view therefore never separates a row from its reverse partner.
assert clean.reverse_pair_id.dropna().isin(clean.instance_id).all()
print("expected: 16 rows flagged (8 phrase pairs), 644 in the clean view (flagged, not deleted). "
      'The notes say 15 / 645: the extra row is "A close-up view" -> "Close up", which train '
      'has as "A close up view" -> "Close up"')

# Do train and dev give the same answer for the phrase pairs they share?
# After Part 5 every train group has exactly one answer.
train_answer = labels_in_key_order(train).groupby(train.group).first()
shared = dev[dev.in_train]
differs = labels_in_key_order(shared) != shared.group.map(train_answer)
print(f"shared dev rows whose label differs from train: {int(differs.sum())}")
print(shared.loc[differs, ["text1", "text2", "label"]].to_string(index=False))
print('expected: 1. "A close-up view" -> "Close up" is NEGATIVE_OTHER in dev but '
      "FORWARD_ENTAILMENT in train. Dev labels are never changed; the row is flagged, so the "
      "clean view leaves it out")


# %% Part 7: grouped, stratified train/validation split
print("\n=== Part 7: grouped, stratified train/validation split ===")
# Grouped: all rows of a phrase pair (both directions and every copy) land on
# the same side, so validation never holds a pair the model trained on.
# Stratified: both sides keep about the same label mix.
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_idx, val_idx = next(sgkf.split(train, train.label, groups=train.group))
train_part = train.iloc[train_idx].reset_index(drop=True)
val = train.iloc[val_idx].reset_index(drop=True)

assert not set(train_part.group) & set(val.group), "a phrase pair is in both train_part and val"
assert val.reverse_pair_id.dropna().isin(val.instance_id).all(), "a val row's reverse is not in val"
print("leak checks passed: no phrase pair on both sides; every val row's reverse is also in val")
print(f"train_part: {len(train_part):,} rows; val: {len(val):,} rows ({len(val) / len(train):.1%} of train)")
print(f"val labels: {counts(val.label)}")
print("notes expect (seed 42, scikit-learn 1.9.1): 2,603 / 436 rows (14.3%); val labels "
      "EQUIVALENCE 116, FORWARD 126, BACKWARD 126, NEGATIVE_OTHER 68")


# %% Part 8: encode labels as numbers
print("\n=== Part 8: encode labels as numbers ===")
for df in (train_part, val, dev):
    df["label_id"] = df.label.map(LABEL2ID)
print(", ".join(f"{i} = {label}" for label, i in LABEL2ID.items()))


# %% Part 9: save the split and the processed files
print("\n=== Part 9: save the split and the processed files ===")
# Which train row went where: IDs only, no text, so this file is committed.
# Any later run, on any computer, must reproduce exactly this split.
assignment = pd.concat([
    pd.DataFrame({"instance_id": train_part.instance_id, "split": "train_part"}),
    pd.DataFrame({"instance_id": val.instance_id, "split": "val"}),
    pd.DataFrame({"instance_id": removed.instance_id, "split": "removed_conflict"}),
]).sort_values("instance_id")
if SPLIT_FILE.exists():
    saved = pd.read_csv(SPLIT_FILE, dtype=str)
    if saved.values.tolist() != assignment.values.tolist():
        raise SystemExit(f"This split differs from the saved {SPLIT_FILE.name}. Nothing was saved. "
                         "Check the scikit-learn version (requirements.txt pins 1.9.1).")
    print(f"split matches the saved data/splits/{SPLIT_FILE.name}")
else:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(SPLIT_FILE, index=False, lineterminator="\n")
    print(f"saved data/splits/{SPLIT_FILE.name}: {len(assignment):,} rows (commit this file)")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
columns = REFERENCE_COLUMNS + ["label_id", "group"]
outputs = {
    "train_part.csv": train_part[columns],
    "val.csv": val[columns],
    "test.csv": dev[columns + ["in_train"]],
    # The official 8 columns, so the official scorer can score validation.
    "val_reference.csv": val[REFERENCE_COLUMNS],
}
for name, df in outputs.items():
    df.to_csv(PROCESSED_DIR / name, index=False, lineterminator="\n")
    print(f"saved data/processed/{name}: {len(df):,} rows")
print("\nDone.")

# %%
