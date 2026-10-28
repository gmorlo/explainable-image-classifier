"""One-time final evaluation of a trained run on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import CLASS_NAMES, RUNS_DIR
from sortwise.data.dataset import TrashNetDataset
from sortwise.inference import load_predictor
from sortwise.training.trainer import evaluate, get_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUNS_DIR / "final" / "efficientnet_b0")
    args = parser.parse_args()

    run_dir = args.run
    device = get_device()
    model = load_predictor(run_dir, device).model
    test_ds = TrashNetDataset("test")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    m = evaluate(model, loader, nn.CrossEntropyLoss(), device)

    cm = confusion_matrix(m["y_true"], m["y_pred"], labels=list(range(len(CLASS_NAMES))))
    # shuffle=False keeps loader order == dataset.samples order, so index i is the same image.
    errors = [
        {"path": test_ds.samples[i][0].relative_to(test_ds.dataset_dir).as_posix(),
         "true": CLASS_NAMES[t], "pred": CLASS_NAMES[p]}
        for i, (t, p) in enumerate(zip(m["y_true"], m["y_pred"]))
        if t != p
    ]

    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "n": len(test_ds),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "loss": m["loss"],
                "per_class": m["per_class"],
                "confusion_matrix": cm.tolist(),
                "errors": errors,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "test_report.txt").write_text(m["report"], encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Test confusion matrix — macro-F1 {m['macro_f1']:.3f}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(run_dir / "test_confusion_matrix.png", dpi=150)

    print(m["report"])
    print(f"accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} errors={len(errors)}/{len(test_ds)}")
    print(cm)
    print(f"Wrote {run_dir / 'test_metrics.json'} and test_confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
