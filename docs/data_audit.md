# Data audit

Audit of the DiCo-NLI English Track 1 reference files (organisers' commit `2c3e909`, see [data/README.md](../data/README.md)) and the train/validation/test files made from them. Every number below is printed by `src/prepare_data.py` (Python 3.14, pandas 3.1.1, scikit-learn 1.9.1).

## Summary

| Check | Result | Action |
|---|---|---|
| Files and columns | train 3,042 rows, dev 660 rows; the 8 official columns in both | None |
| Missing values | Only `reverse_pair_id`: 478 train and 106 dev rows, exactly the NEGATIVE_OTHER rows, which have no reversed partner | Expected; none |
| Reverse links | All 2,564 train and 554 dev reversible rows point to a partner that points back, has the two phrases swapped and has the reversed label | None |
| Text cleaning needed | No empty phrases, no leading, trailing or double spaces. Only English, no unknown labels, no duplicate `instance_id`s | None; text kept exactly as written |
| Duplicate pairs (train) | 51 extra exact copies (94 rows); 68 extra copies (126 rows) when capitals and punctuation are ignored | Kept; the group key keeps every copy on the same side of the split |
| Label conflicts (train) | 1 phrase pair: "South Africans" → "South Africa" is labelled both FORWARD_ENTAILMENT and NEGATIVE_OTHER (3 rows) | 3 rows removed, 3,039 left |
| Train/dev overlap | 16 dev rows (8 phrase pairs) share a group with train; 1 of them has a different label in dev | Flagged with `in_train`, not deleted; clean test view of 644 rows |
| Validation split | 2,603 / 436 rows (14.3% validation); 0 groups on both sides; every validation row's reverse is also in validation | Grouped and stratified, seed 42 |

## Group key

A phrase pair, its reverse and any copies must stay together. Each row gets a key built from its two phrases: lowercased, punctuation turned into spaces, repeated spaces squeezed, sorted, and joined with `||`. For example, "Conservative opposition" + "Conservatives" gives `conservative opposition || conservatives`, and the reversed row gives the same key. The key is used only for comparing; the saved phrases are unchanged.

The train file has 1,724 groups but 1,760 `pair_id`s, because the same phrase pair sometimes appears again under a new `pair_id`.

Ignoring punctuation, not only capitals, merges exactly 2 groups. Both are true copies:

| File | Phrase pair | Label |
|---|---|---|
| train | "for gang-rape" → "for rape" and "for gang rape" → "for rape" | FORWARD_ENTAILMENT (both) |
| dev / train | "A close-up view" → "Close up" / "A close up view" → "Close up" | NEGATIVE_OTHER / FORWARD_ENTAILMENT |

A key that ignores only capitals gives 1,725 groups and finds 15 overlapping dev rows (645 in the clean view). It keeps both pairs apart: the two "gang-rape" copies could land on different sides of the split, and the "close-up" dev row would count as unseen although train has the same phrases.

## Label conflict

Labels are compared after rewriting each one for the direction of its group key, so (A, B) = FORWARD and its reverse (B, A) = BACKWARD count as the same answer. One group still has two answers:

| text1 | text2 | label |
|---|---|---|
| South Africans | South Africa | FORWARD_ENTAILMENT |
| South Africa | South Africans | BACKWARD_ENTAILMENT |
| South Africans | South Africa | NEGATIVE_OTHER |

The data does not show which answer is right, so all 3 rows are removed rather than one label being chosen. Two different correct answers for the same phrases would teach the model a contradiction. The split file records the 3 rows as `removed_conflict`.

## Train/dev overlap

Dev is the test set, so it is never changed and is not used for any decision. Dev rows whose group also appears in train are flagged with `in_train = True`. Scores can then be reported on the full test set (official) and on the clean view (phrase pairs never seen in training).

| Dev phrase pair | Dev rows | Label in dev |
|---|---|---|
| US stocks / Stocks | 4 | FORWARD / BACKWARD |
| Suicide bombers / Suicide bomber | 2 | FORWARD / BACKWARD |
| a small baby / a baby | 2 | FORWARD / BACKWARD |
| driving down / driving | 2 | FORWARD / BACKWARD |
| A black and white horned cow / a black and white cow | 2 | FORWARD / BACKWARD |
| play / are playing | 2 | EQUIVALENCE |
| into the camera → at the camera | 1 | NEGATIVE_OTHER |
| A close-up view → Close up | 1 | NEGATIVE_OTHER (train: FORWARD_ENTAILMENT) |

A pair and its reverse share a group, so they are always flagged together. The clean view never holds a row without its reverse partner.

## Validation split

Validation is taken from train with `StratifiedGroupKFold` (7 folds, shuffled, seed 42); the first fold is validation.

- **Grouped:** all rows of a phrase pair, in both directions and every copy, land on the same side, so validation never holds a pair the model trained on.
- **Stratified:** both sides keep about the same label mix.

| Split | Rows | Groups | Reversible pairs | NEGATIVE_OTHER rows |
|---|---|---|---|---|
| train after removal | 3,039 | 1,723 | 1,281 | 477 |
| train_part | 2,603 | 1,476 | 1,097 | 409 |
| val | 436 | 247 | 184 | 68 |
| test (dev) | 660 | 381 | 277 | 106 |
| test, clean view | 644 | 373 | 270 | 104 |

| Split | EQUIVALENCE | FORWARD_ENTAILMENT | BACKWARD_ENTAILMENT | NEGATIVE_OTHER |
|---|---|---|---|---|
| train after removal | 802 (26.4%) | 880 (29.0%) | 880 (29.0%) | 477 (15.7%) |
| train_part | 686 (26.4%) | 754 (29.0%) | 754 (29.0%) | 409 (15.7%) |
| val | 116 (26.6%) | 126 (28.9%) | 126 (28.9%) | 68 (15.6%) |
| test (dev) | 174 (26.4%) | 190 (28.8%) | 190 (28.8%) | 106 (16.1%) |

Two checks stop the script if the split leaks: no group may be in both `train_part` and `val`, and every validation row's reverse partner must also be in validation. Both pass.

Another scikit-learn version may shuffle differently, so `requirements.txt` pins 1.9.1. The split is saved as IDs only in `data/splits/train_val_split.csv`, which is committed. Every later run must reproduce it exactly, or `prepare_data.py` stops without saving.

## What is deliberately not done

- No lowercasing, punctuation removal, stopword removal, stemming or lemmatising in the saved files. "into" vs "at" or "bomber" vs "bombers" can be exactly what decides the label.
- No "fixing" of headline fragments such as "'s opposition party".
- Labels are encoded in one fixed order: 0 = EQUIVALENCE, 1 = FORWARD_ENTAILMENT, 2 = BACKWARD_ENTAILMENT, 3 = NEGATIVE_OTHER.
- Models get only `text1` and `text2`. NEGATIVE_OTHER rows are the only rows without a `reverse_pair_id`, so the ID columns would give the label away.

## Output files

| File | Rows | Contents | Committed |
|---|---|---|---|
| `data/splits/train_val_split.csv` | 3,042 | `instance_id` and side: 2,603 `train_part`, 436 `val`, 3 `removed_conflict` | Yes |
| `data/processed/train_part.csv` | 2,603 | 8 official columns + `label_id`, `group` | No |
| `data/processed/val.csv` | 436 | 8 official columns + `label_id`, `group` | No |
| `data/processed/test.csv` | 660 | Dev: 8 official columns + `label_id`, `group`, `in_train` | No |
| `data/processed/val_reference.csv` | 436 | Validation in the official 8-column format, for the official scorer | No |

To reproduce: `python src/download_data.py`, then `python src/prepare_data.py` from the project root.
