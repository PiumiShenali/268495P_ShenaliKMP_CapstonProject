# 268495P_ShenaliKMP_CapstonProject
Capstone project repository - NLP Project

**Improving Natural Language Inference by Understanding Directional Relationships in Phrase Pairs**

Capstone Project, based on SemEval-2027 Task 2 (DiCo-NLI), English track only.

## Setup

Python 3.14 on Windows (PowerShell), from the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
```

The last command downloads the English data into `data/raw/` and checks each file's SHA-256 checksum. See [data/README.md](data/README.md).

## Folders

| Folder | Contents |
|---|---|
| `data/raw/` | Official data, downloaded by `src/download_data.py` (not committed) |
| `data/processed/` | Training, validation and test files made by our scripts |
| `src/` | Python code |
| `notebooks/` | EDA and analysis notebooks |
| `configs/` | Experiment settings |
| `results/` | Scores, figures and predictions |
| `docs/` | Data audit and other write-ups |
| `app/` | Streamlit demo |
| `tests/` | Unit tests |
