"""Compare latest head vs finetune ResNet18 runs on validation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import COMPARISONS_DIR, RUNS_DIR


def _load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run(variant: str) -> Path:
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(f"No runs directory: {RUNS_DIR}")
    candidates = [
        p
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / "metrics.json").is_file()
    ]
    matched = []
    for p in candidates:
        metrics = _load_metrics(p)
        if metrics.get("variant") == variant:
            matched.append(p)
    if not matched:
        raise FileNotFoundError(f"No completed runs for variant={variant!r} under {RUNS_DIR}")
    return max(matched, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-run", type=Path, default=None)
    parser.add_argument("--finetune-run", type=Path, default=None)
    args = parser.parse_args()

    head_dir = args.head_run or _latest_run("head")
    ft_dir = args.finetune_run or _latest_run("finetune")
    head = _load_metrics(head_dir)
    finetune = _load_metrics(ft_dir)

    winner = (
        "finetune"
        if finetune["best_val_macro_f1"] > head["best_val_macro_f1"]
        else "head"
        if head["best_val_macro_f1"] > finetune["best_val_macro_f1"]
        else "tie"
    )

    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "best_val_macro_f1",
        "winner": winner,
        "runs": {
            "head": {
                "run_dir": str(head_dir),
                **{k: head[k] for k in (
                    "run_id",
                    "best_epoch",
                    "best_val_macro_f1",
                    "best_val_accuracy",
                    "train_seconds",
                    "weights_bytes",
                    "trainable_parameters",
                    "device",
                    "gpu_name",
                ) if k in head},
            },
            "finetune": {
                "run_dir": str(ft_dir),
                **{k: finetune[k] for k in (
                    "run_id",
                    "best_epoch",
                    "best_val_macro_f1",
                    "best_val_accuracy",
                    "train_seconds",
                    "weights_bytes",
                    "trainable_parameters",
                    "device",
                    "gpu_name",
                ) if k in finetune},
            },
        },
        "notes": [
            "Comparison uses validation metrics only; test set was not used.",
            "Winner chosen by higher best_val_macro_f1.",
        ],
    }

    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = COMPARISONS_DIR / "resnet18_head_vs_finetune.json"
    out_md = COMPARISONS_DIR / "resnet18_head_vs_finetune.md"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    h = comparison["runs"]["head"]
    f = comparison["runs"]["finetune"]
    md = "\n".join(
        [
            "# ResNet18: head vs finetune (validation)",
            "",
            f"Winner: **{winner}** (by val macro-F1)",
            "",
            "| Variant | Val macro-F1 | Val accuracy | Best epoch | Train seconds | Trainable params | Weights bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| head | {h['best_val_macro_f1']:.4f} | {h['best_val_accuracy']:.4f} | "
                f"{h['best_epoch']} | {h['train_seconds']:.1f} | {h['trainable_parameters']} | "
                f"{h['weights_bytes']} |"
            ),
            (
                f"| finetune | {f['best_val_macro_f1']:.4f} | {f['best_val_accuracy']:.4f} | "
                f"{f['best_epoch']} | {f['train_seconds']:.1f} | {f['trainable_parameters']} | "
                f"{f['weights_bytes']} |"
            ),
            "",
            f"- head run: `{h.get('run_id')}`",
            f"- finetune run: `{f.get('run_id')}`",
            "- Test set not used for this comparison.",
            "",
        ]
    )
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
