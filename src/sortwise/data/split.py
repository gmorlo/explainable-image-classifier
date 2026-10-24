"""Stratified train/val/test split with a fixed seed."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from sortwise.config import SPLIT_RATIOS, SPLIT_SEED, SPLITS_DIR
from sortwise.data.manifest import load_manifest


def create_stratified_split(
    manifest: dict[str, Any] | None = None,
    *,
    ratios: dict[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Return mapping split_name -> list of file ids (relative paths)."""
    data = manifest if manifest is not None else load_manifest()
    ratios = ratios or SPLIT_RATIOS
    seed = SPLIT_SEED if seed is None else seed

    usable = [f for f in data["files"] if f.get("ok", True)]
    if not usable:
        raise ValueError("No usable files in manifest for splitting")

    ids = [f["id"] for f in usable]
    labels = [f["class"] for f in usable]

    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    ids_train, ids_temp, y_train, y_temp = train_test_split(
        ids,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    relative_test = test_ratio / (val_ratio + test_ratio)
    ids_val, ids_test, _, _ = train_test_split(
        ids_temp,
        y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "train": sorted(ids_train),
        "val": sorted(ids_val),
        "test": sorted(ids_test),
    }


def _counts_by_class(file_ids: list[str], id_to_class: dict[str, str]) -> dict[str, int]:
    return dict(Counter(id_to_class[i] for i in file_ids))


def save_splits(
    splits: dict[str, list[str]],
    manifest: dict[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
    seed: int | None = None,
    ratios: dict[str, float] | None = None,
) -> Path:
    """Write splits.json plus per-split id lists under data/processed/splits/."""
    data = manifest if manifest is not None else load_manifest()
    out_dir = output_dir or SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_class = {f["id"]: f["class"] for f in data["files"]}
    seed = SPLIT_SEED if seed is None else seed
    ratios = ratios or SPLIT_RATIOS

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": ratios,
        "counts": {name: len(ids) for name, ids in splits.items()},
        "class_counts": {
            name: _counts_by_class(ids, id_to_class) for name, ids in splits.items()
        },
        "splits": splits,
    }

    out_path = out_dir / "splits.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, ids in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    return out_path
