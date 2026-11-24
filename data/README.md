# Data

This project uses only the **English track (Track 1)** of the SemEval-2027 Task 2 (DiCo-NLI) data.

| | |
|---|---|
| Source | <https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI>, folder `final_data/` |
| Pinned version | commit `2c3e909bfa263a583976d30e2f3fcfc43fedff20` (12 Nov 2026) |
| How to get it | `python src/download_data.py` saves the files to `data/raw/` |

The organizers' repository is still being updated, so the data is fixed to one commit and every file is checked with SHA-256. The data itself is not stored in this repository; `data/raw/` is in `.gitignore`.

## Files

| File | Rows | SHA-256 |
|---|---|---|
| `dico_nli_train_track1_participant_labeled.csv` | 3,042 | `5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211` |
| `dico_nli_train_track1_reference.csv` | 3,042 | `c8f4d8587711e0beaefc115606e5c29c3e78dc6660e074967428af0fed069ee4` |
| `dico_nli_dev_track1_participant_labeled.csv` | 660 | `80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511` |
| `dico_nli_dev_track1_reference.csv` | 660 | `8d5cd11c1c60df3c9b59254d12a60fb2cf43e91dde01b46c8a8f296b806134bc` |

Columns: `instance_id, pair_id, text1_lang, text2_lang, text1, text2, label`. The `reference` files add `reverse_pair_id`, which links each row to its reversed partner and is needed by the official scorer.

Models must never see `instance_id`, `pair_id` or `reverse_pair_id`. NEGATIVE_OTHER rows are the only rows without a reversed partner, so these columns would give the label away.

## How the project uses them

| Data | Role |
|---|---|
| train | Split into a training part and a validation part, grouped by phrase pair (step 2) |
| dev | Held out as the **test set** and used once at the end, because the official test set is not released before the capstone deadline |
| trial (organizers' `trial_data/`) | Not used: it is an older preview, and 495 of its 797 rows also appear in train or dev |

## Licence

The organizers' repository is released under GPL-3.0.
