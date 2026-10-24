"""Project paths and data-prep defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
SPLITS_PATH = SPLITS_DIR / "splits.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
COMPARISONS_DIR = ARTIFACTS_DIR / "comparisons"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATASET_NAME = "dataset-resized"
DATASET_DIR = RAW_DIR / DATASET_NAME
HF_DATASET_REPO = "garythung/trashnet"
HF_ZIP_FILENAME = "dataset-resized.zip"

CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

CLASS_LABELS_PL = {
    "cardboard": "Karton",
    "glass": "Szkło",
    "metal": "Metal",
    "paper": "Papier",
    "plastic": "Plastik",
    "trash": "Pozostałe odpady",
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
