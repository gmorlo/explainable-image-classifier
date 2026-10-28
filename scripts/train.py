"""Train model variants (head-only or last-stage fine-tune) on TrashNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import CONFIGS_DIR
from sortwise.data.dataset import create_dataloaders
from sortwise.models.factory import build_model
from sortwise.training.trainer import TrainConfig, train_model


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_variant(cfg_doc: dict, variant: str) -> dict:
    if variant not in cfg_doc["variants"]:
        raise SystemExit(f"Unknown variant {variant!r}. Choose from: {list(cfg_doc['variants'])}")

    v = cfg_doc["variants"][variant]
    train_cfg = TrainConfig(
        variant=variant,
        trainable=v["trainable"],
        lr=float(v["lr"]),
        batch_size=int(cfg_doc["batch_size"]),
        max_epochs=int(cfg_doc["max_epochs"]),
        patience=int(cfg_doc["patience"]),
        weight_decay=float(cfg_doc["weight_decay"]),
        seed=int(cfg_doc["seed"]),
        num_classes=int(cfg_doc["num_classes"]),
        num_workers=int(cfg_doc.get("num_workers", 4)),
        model_name=str(cfg_doc.get("model", "resnet18")),
        optimizer_name=str(cfg_doc.get("optimizer", "adam")),
    )

    loaders = create_dataloaders(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )
    model = build_model(
        train_cfg.model_name,
        num_classes=train_cfg.num_classes,
        trainable=train_cfg.trainable,
    )
    snapshot = {
        "yaml": cfg_doc,
        "variant": variant,
        "resolved": {
            "lr": train_cfg.lr,
            "trainable": train_cfg.trainable,
        },
    }
    print(f"=== Training model={train_cfg.model_name} variant={variant} trainable={train_cfg.trainable} lr={train_cfg.lr} ===")
    result = train_model(model, loaders, train_cfg, config_snapshot=snapshot)
    print(f"Done. run_dir={result['run_dir']}")
    print(
        f"best_val_macro_f1={result['metrics']['best_val_macro_f1']:.4f} "
        f"best_epoch={result['metrics']['best_epoch']}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "train_resnet18.yaml",
        help="Path to training YAML",
    )
    parser.add_argument(
        "--variant",
        choices=["head", "finetune", "both"],
        default="both",
        help="Which variant to train",
    )
    args = parser.parse_args()

    cfg_doc = load_yaml(args.config)
    variants = ["head", "finetune"] if args.variant == "both" else [args.variant]
    results = []
    for variant in variants:
        results.append(run_variant(cfg_doc, variant))

    if len(results) == 2:
        # Defer comparison to compare_variants.py using latest runs;
        # still print a quick summary here.
        print("\n=== Quick summary ===")
        for r in results:
            m = r["metrics"]
            print(
                f"{m['variant']}: macro_f1={m['best_val_macro_f1']:.4f} "
                f"acc={m['best_val_accuracy']:.4f} "
                f"seconds={m['train_seconds']:.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
