"""Download the English (Track 1) DiCo-NLI data at a fixed version and check it.

The organizers' repository is still being updated, so we pin one commit and
check every file against its SHA-256 checksum. The script stops with an error
if any file differs, so every run of the project uses exactly the same data.

Usage (from the project root):
    python src/download_data.py                    # download from GitHub
    python src/download_data.py --from-local DIR   # copy from a local copy of the
                                                   # organizers' repo (DIR contains final_data/)
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI"
# 12 Nov 2026: "Clarify PhrasIS training policy and contamination in FAQ"
COMMIT = "2c3e909bfa263a583976d30e2f3fcfc43fedff20"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/final_data"

# Path inside the organizers' final_data/ folder -> expected SHA-256.
FILES = {
    "train/dico_nli_train_track1_participant_labeled.csv": "5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211",
    "train/dico_nli_train_track1_reference.csv": "c8f4d8587711e0beaefc115606e5c29c3e78dc6660e074967428af0fed069ee4",
    "dev/dico_nli_dev_track1_participant_labeled.csv": "80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511",
    "dev/dico_nli_dev_track1_reference.csv": "8d5cd11c1c60df3c9b59254d12a60fb2cf43e91dde01b46c8a8f296b806134bc",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel, target, local_root):
    if local_root is not None:
        shutil.copyfile(local_root / "final_data" / rel, target)
    else:
        with urllib.request.urlopen(f"{BASE_URL}/{rel}", timeout=60) as resp, open(target, "wb") as out:
            shutil.copyfileobj(resp, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-local", type=Path, metavar="DIR",
                        help="root of a local copy of the organizers' repo (contains final_data/)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bad = []
    for rel, expected in FILES.items():
        target = RAW_DIR / Path(rel).name
        fetch(rel, target, args.from_local)
        if sha256(target) == expected:
            print(f"OK   data/raw/{target.name}")
        else:
            print(f"BAD  data/raw/{target.name}")
            target.unlink()  # never leave a wrong file where scripts could read it
            bad.append(target.name)

    if bad:
        sys.exit(f"Checksum mismatch for {bad}: these files differ from commit {COMMIT[:7]}.")
    print(f"All {len(FILES)} files match commit {COMMIT[:7]}.")


if __name__ == "__main__":
    main()
