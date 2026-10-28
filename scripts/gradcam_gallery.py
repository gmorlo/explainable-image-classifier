"""Grad-CAM galleries for the final model: one correct example per class and the test errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import CLASS_NAMES, DATASET_DIR, RUNS_DIR
from sortwise.explain.gradcam import grad_cam, overlay, preprocess
from sortwise.models.factory import MODEL_SPECS, build_model
from sortwise.training.trainer import get_device


def draw(model, layer, device, items: list[dict], title: str, out: Path, cols: int = 4) -> None:
    """Each item: path, true. Shows image + heatmap for the predicted class."""
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.2, rows * 2.3))
    axes = axes.reshape(rows, cols * 2)
    for ax in axes.ravel():
        ax.axis("off")
    for k, item in enumerate(items):
        x, crop = preprocess(Image.open(DATASET_DIR / item["path"]))
        cam, probs = grad_cam(model, x.to(device), layer)
        pred = int(probs.argmax())
        ok = CLASS_NAMES[pred] == item["true"]
        r, c = divmod(k, cols)
        axes[r, 2 * c].imshow(crop)
        axes[r, 2 * c].set_title(f"true: {item['true']}", fontsize=9)
        axes[r, 2 * c + 1].imshow(overlay(crop, cam))
        axes[r, 2 * c + 1].set_title(
            f"pred: {CLASS_NAMES[pred]} ({probs[pred]:.0%})", fontsize=9, color="green" if ok else "red"
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUNS_DIR / "final" / "efficientnet_b0")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "img")
    parser.add_argument("--max-errors", type=int, default=12)
    args = parser.parse_args()

    train_cfg = json.loads((args.run / "train_config.json").read_text(encoding="utf-8"))
    test = json.loads((args.run / "test_metrics.json").read_text(encoding="utf-8"))
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    model_name = train_cfg["model_name"]
    model = build_model(model_name, trainable=ckpt["trainable"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = get_device()
    model.to(device)
    layer = MODEL_SPECS[model_name].cam_layer(model)

    error_paths = {e["path"] for e in test["errors"]}
    splits = json.loads((ROOT / "data/processed/splits/splits.json").read_text(encoding="utf-8"))
    correct = []
    for cls in CLASS_NAMES:
        path = next(p for p in splits["splits"]["test"] if p.startswith(cls + "/") and p not in error_paths)
        correct.append({"path": path, "true": cls})
    # Most common confusions first (cardboard->paper, plastic->glass, ...).
    pair_count: dict[tuple, int] = {}
    for e in test["errors"]:
        pair_count[(e["true"], e["pred"])] = pair_count.get((e["true"], e["pred"]), 0) + 1
    errors = sorted(test["errors"], key=lambda e: -pair_count[(e["true"], e["pred"])])[: args.max_errors]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw(model, layer, device, correct, "Grad-CAM — correct test predictions", args.out_dir / "gradcam_correct.png", cols=3)
    draw(model, layer, device, errors, "Grad-CAM — test errors (most common confusions first)",
         args.out_dir / "gradcam_errors.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
