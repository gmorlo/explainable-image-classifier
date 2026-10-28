"""Charts for README / docs from the saved run artifacts, plus the same numbers as docs/data/results.json."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import CLASS_LABELS, COMPARISONS_DIR, RUNS_DIR

OUT_IMG = ROOT / "docs" / "img"
OUT_DATA = ROOT / "docs" / "data" / "results.json"
FINAL_RUN = RUNS_DIR / "final" / "efficientnet_b0"
MODEL_LABELS = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet18", "mobilenet_v3_small": "MobileNetV3-Small"}

# Validated categorical slots 1-2 (dataviz reference palette, light surface) + text/grid tokens.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finish(fig, ax, title: str, subtitle: str, name: str) -> None:
    # Offsets in points so title and subtitle never overlap, whatever the figure height.
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points",
                fontsize=15, fontweight="bold", color=INK, va="bottom")
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 12), textcoords="offset points",
                fontsize=11, color=INK_2, va="bottom")
    fig.savefig(OUT_IMG / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {OUT_IMG / name}")


def model_comparison(rows: list[dict]) -> None:
    order = sorted({r["model"] for r in rows}, key=lambda m: -max(r["best_val_macro_f1"] for r in rows if r["model"] == m))
    by = {(r["model"], r["variant"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for y, m in enumerate(reversed(order)):
        head, fine = by[(m, "head")]["best_val_macro_f1"], by[(m, "finetune")]["best_val_macro_f1"]
        ax.plot([head, fine], [y, y], color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
        ax.scatter([head], [y], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="Head only" if y == 0 else None)
        ax.scatter([fine], [y], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="Fine-tune last stage" if y == 0 else None)
        ax.annotate(f"{fine:.3f}", (fine, y), xytext=(10, 0), textcoords="offset points", va="center", color=INK)
        ax.annotate(f"{head:.3f}", (head, y), xytext=(-10, 0), textcoords="offset points", va="center", ha="right", color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{MODEL_LABELS[m]}  ·  {by[(m, 'finetune')]['weights_mb']:.0f} MB" for m in reversed(order)])
    ax.set_xlim(0.72, 0.96)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation macro-F1")
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.2), fontsize=10)
    finish(fig, ax, "Fine-tuning the last stage adds +0.06–0.13", "3 backbones × 2 training modes, same split and preprocessing", "chart_models.png")


def seed_spread(rows: list[dict]) -> list[dict]:
    groups: dict[float, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["lr"], []).append(r)
    settings = sorted(groups, key=lambda lr: -statistics.mean(r["best_val_macro_f1"] for r in groups[lr]))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    out = []
    for y, lr in enumerate(reversed(settings)):
        f1s = [r["best_val_macro_f1"] for r in groups[lr]]
        mean, std = statistics.mean(f1s), statistics.stdev(f1s)
        out.append({"lr": lr, "seeds": {r["seed"]: r["best_val_macro_f1"] for r in groups[lr]}, "mean": mean, "std": std})
        color = ORANGE if lr == settings[0] else BLUE
        ax.plot([min(f1s), max(f1s)], [y, y], color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
        ax.scatter(f1s, [y] * len(f1s), s=80, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.plot([mean, mean], [y - 0.22, y + 0.22], color=INK, linewidth=2, zorder=4)
        ax.annotate(f"mean {mean:.3f} ± {std:.3f}", (max(f1s), y), xytext=(14, 0), textcoords="offset points", va="center", color=INK)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels([f"lr {lr:g}" + ("  (chosen)" if lr == settings[0] else "") for lr in reversed(settings)])
    ax.set_xlim(0.875, 0.965)
    ax.set_ylim(-0.6, len(settings) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation macro-F1 (dots: seeds 42, 1, 2 · bar: mean)")
    finish(fig, ax, "Seed noise is bigger than the gap between settings",
           "EfficientNet-B0 fine-tune, batch 16, top-2 learning rates rerun on 3 seeds", "chart_seeds.png")
    return out


def per_class(test: dict) -> None:
    items = sorted(test["per_class"].items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ys = range(len(items))
    ax.barh(ys, [v["f1"] for _, v in items], height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for y, (_, v) in zip(ys, items):
        ax.annotate(f"{v['f1']:.2f}", (v["f1"], y), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{CLASS_LABELS[k]}  (n={v['support']})" for k, v in items])
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test F1")
    finish(fig, ax, f"Test set: macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}",
           "Per-class F1 on the held-out TrashNet test split (used once)", "chart_per_class.png")


def training_curve(history: list[dict]) -> None:
    epochs = [h["epoch"] for h in history]
    f1 = [h["val_macro_f1"] for h in history]
    best = max(history, key=lambda h: h["val_macro_f1"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(epochs, f1, color=BLUE, linewidth=2)
    ax.scatter([best["epoch"]], [best["val_macro_f1"]], s=80, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"best: epoch {best['epoch']}, {best['val_macro_f1']:.3f} → saved checkpoint",
                (best["epoch"], best["val_macro_f1"]), xytext=(0, 14), textcoords="offset points", ha="center", color=INK)
    ax.annotate(f"stopped at epoch {epochs[-1]}: no gain for {epochs[-1] - best['epoch']} epochs", (epochs[-1], 0.785),
                ha="right", va="bottom", color=INK_2)
    ax.set_xlim(0.5, epochs[-1] + 0.5)
    ax.set_ylim(0.78, 0.95)
    ax.set_xticks(epochs[::2])
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("Epoch")
    finish(fig, ax, "Early stopping on validation macro-F1", "Final EfficientNet-B0 run: lr 3e-4, batch 16, seed 42",
           "chart_training.png")


def main() -> int:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    models = load(COMPARISONS_DIR / "model_comparison.json")["rows"]
    # The seed-repeat grid is the newest one with more than one seed.
    grids = [load(p) for p in sorted(COMPARISONS_DIR.glob("grid_*.json"))]
    seed_rows = [g for g in grids if len({r.get("seed") for r in g["rows"]}) > 1][-1]["rows"]
    test = load(FINAL_RUN / "test_metrics.json")
    history = load(FINAL_RUN / "history.json")

    model_comparison(models)
    seeds = seed_spread(seed_rows)
    per_class(test)
    training_curve(history)

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps({
        "models_validation": [{k: r[k] for k in ("model", "variant", "best_val_macro_f1", "best_val_accuracy", "weights_mb")}
                              for r in models],
        "seed_repeats_validation": seeds,
        "test": {k: test[k] for k in ("n", "accuracy", "macro_f1", "per_class", "confusion_matrix")},
        "class_order": list(CLASS_LABELS),
        "training_history": history,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
