# 268495P_ShenaliKMP_CapstonProject
Capstone project repository - NLP Project

**Improving Natural Language Inference by Understanding Directional Relationships in Phrase Pairs**

Capstone Project, based on SemEval-2027 Task 2 (DiCo-NLI), English track only.

Data: the official DiCo-NLI English Track 1 files, pinned to the organisers' commit `2c3e909` (12 Nov 2026) of [ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI). See [data/README.md](data/README.md).

## Setup

Python 3.14 on Windows (PowerShell), from the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py
```

`download_data.py` downloads the English data into `data/raw/` and checks each file's SHA-256 checksum (see [data/README.md](data/README.md)). `prepare_data.py` audits the data, removes the one conflicting phrase pair, and makes the train/validation/test files in `data/processed/`.

## Folders

| Folder | Contents |
|---|---|
| `data/raw/` | Official data, downloaded by `src/download_data.py` (not committed) |
| `data/processed/` | Training, validation and test files made by `src/prepare_data.py` (not committed) |
| `data/splits/` | Which training row went to training or validation (IDs only, committed) |
| `src/` | Python code |
| `notebooks/` | EDA and analysis notebooks |
| `configs/` | Experiment settings |
| `results/` | Scores, figures and predictions |
| `docs/` | Data audit and other write-ups |
| `app/` | Streamlit demo |
| `tests/` | Unit tests |
