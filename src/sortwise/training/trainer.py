"""Train/eval loop with early stopping on validation macro-F1."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sortwise.config import CLASS_NAMES, RUNS_DIR
from sortwise.models.resnet import count_trainable_parameters
from sortwise.training.metrics import classification_metrics


@dataclass
class TrainConfig:
    variant: str
    trainable: str
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    weight_decay: float
    seed: int
    num_classes: int = 6
    num_workers: int = 4
    model_name: str = "resnet18"
    optimizer_name: str = "adam"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += float(loss.item())
        n_batches += 1
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    *,
    config_snapshot: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer_name.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_name or f"resnet18_{cfg.variant}_{stamp}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "class_names": list(CLASS_NAMES),
        "class_to_idx": {name: i for i, name in enumerate(CLASS_NAMES)},
    }
    (run_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    if config_snapshot is not None:
        (run_dir / "config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2), encoding="utf-8"
        )
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    best_path = run_dir / "best.pt"
    t0 = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{cfg.max_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": cfg.variant,
                    "trainable": cfg.trainable,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "class_names": list(CLASS_NAMES),
                },
                best_path,
            )
            (run_dir / "best_val_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "accuracy": val_metrics["accuracy"],
                        "macro_f1": val_metrics["macro_f1"],
                        "loss": val_metrics["loss"],
                        "per_class": val_metrics["per_class"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "best_val_report.txt").write_text(
                val_metrics["report"], encoding="utf-8"
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    train_seconds = time.perf_counter() - t0
    weights_bytes = best_path.stat().st_size if best_path.is_file() else 0

    # Reload best weights for final recorded state
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    metrics = {
        "run_id": run_id,
        "variant": cfg.variant,
        "trainable": cfg.trainable,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_accuracy": next(
            (h["val_accuracy"] for h in history if h["epoch"] == best_epoch), None
        ),
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "weights_bytes": weights_bytes,
        "trainable_parameters": count_trainable_parameters(model),
        "torch_version": torch.__version__,
        **_device_info(device),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {"run_dir": str(run_dir), "metrics": metrics, "history": history}
