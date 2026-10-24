"""Download and extract TrashNet dataset-resized from Hugging Face."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from sortwise.config import (
    DATASET_DIR,
    DATASET_NAME,
    HF_DATASET_REPO,
    HF_ZIP_FILENAME,
    RAW_DIR,
)


def download_trashnet(*, force: bool = False) -> Path:
    """Download dataset-resized.zip, extract under data/raw/, return dataset root."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    marker = DATASET_DIR / ".download_complete"
    if DATASET_DIR.is_dir() and marker.exists() and not force:
        return DATASET_DIR

    if force and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    zip_path = Path(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=HF_ZIP_FILENAME,
            local_dir=str(RAW_DIR),
        )
    )

    extract_root = RAW_DIR
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # Zip may contain dataset-resized/ at top level or nested once.
    if not DATASET_DIR.is_dir():
        candidates = list(RAW_DIR.rglob(DATASET_NAME))
        dirs = [p for p in candidates if p.is_dir()]
        if not dirs:
            raise FileNotFoundError(
                f"After extract, expected directory named {DATASET_NAME!r} under {RAW_DIR}"
            )
        src = dirs[0]
        if src.resolve() != DATASET_DIR.resolve():
            if DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR)
            shutil.move(str(src), str(DATASET_DIR))

    marker.write_text("ok\n", encoding="utf-8")
    return DATASET_DIR
