"""PyTorch dataset over TrashNet using the fixed stratified split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sortwise.config import CLASS_NAMES, DATASET_DIR, SPLITS_PATH
from sortwise.data.transforms import build_eval_transforms, build_train_transforms


class TrashNetDataset(Dataset):
    def __init__(
        self,
        split: str,
        *,
        dataset_dir: Path | None = None,
        splits_path: Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")

        self.split = split
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        splits_file = splits_path or SPLITS_PATH
        if not splits_file.is_file():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        doc = json.loads(splits_file.read_text(encoding="utf-8"))
        rel_paths = doc["splits"][split]
        self.samples: list[tuple[Path, int]] = []
        for rel in rel_paths:
            path = self.dataset_dir / rel
            class_name = Path(rel).parts[0]
            if class_name not in self.class_to_idx:
                raise ValueError(f"Unknown class in split entry {rel!r}")
            self.samples.append((path, self.class_to_idx[class_name]))

        if transform is None:
            transform = (
                build_train_transforms() if split == "train" else build_eval_transforms()
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_dir: Path | None = None,
    splits_path: Path | None = None,
) -> dict[str, DataLoader]:
    """Build train and val loaders only (test reserved for later evaluation)."""
    train_ds = TrashNetDataset(
        "train",
        dataset_dir=dataset_dir,
        splits_path=splits_path,
        transform=build_train_transforms(),
    )
    val_ds = TrashNetDataset(
        "val",
        dataset_dir=dataset_dir,
        splits_path=splits_path,
        transform=build_eval_transforms(),
    )

    pin_memory = True  # effective when tensors land on CUDA in the training loop
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders
