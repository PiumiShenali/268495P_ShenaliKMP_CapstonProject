"""Settings and small helpers shared by all project scripts."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

# Fixed label order: number i always means LABELS[i], in every script and model.
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# What a label becomes when the two phrases are swapped, (A, B) -> (B, A).
# NEGATIVE_OTHER -> NEGATIVE_OTHER is our own design assumption: the official
# metrics leave NEGATIVE_OTHER out of the reversal rules.
REVERSE_LABEL = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER": "NEGATIVE_OTHER",
}

# Columns of the official reference files, in their order.
REFERENCE_COLUMNS = ["instance_id", "pair_id", "text1_lang", "text2_lang",
                     "text1", "text2", "reverse_pair_id", "label"]


def read_reference(path):
    """Read an official CSV with every value kept exactly as written.

    By default pandas turns phrases such as "NA", "null" or "None" into missing
    values. keep_default_na=False stops that, so only empty cells are missing.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def normalise(text):
    """The form of a phrase used for comparing. It is never saved.

    Lowercase, turn punctuation into spaces and squeeze repeated spaces, so
    "A close-up view" and "a close up view" count as the same phrase.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def group_key(text1, text2):
    """One key per phrase pair, ignoring order, capitals and punctuation.

    (A, B), its reverse (B, A) and any copies all get the same key, so a split
    by this key keeps them on the same side.
    """
    return " || ".join(sorted([normalise(text1), normalise(text2)]))
