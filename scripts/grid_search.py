"""Hyperparameter grid (lr x batch size x weight decay x seed) for one model variant."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from train import load_yaml, run_variant

from sortwise.config import COMPARISONS_DIR, CONFIGS_DIR

LRS = (1e-3, 3e-4, 1e-4)
BATCH_SIZES = (16, 32)
WEIGHT_DECAYS = (0.0, 1e-4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "train_efficientnet_b0.yaml")
    parser.add_argument("--variant", default="finetune")
    parser.add_argument("--lrs", type=float, nargs="+", default=LRS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=WEIGHT_DECAYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: seed from config")
    args = parser.parse_args()

    base = load_yaml(args.config)
    seeds = args.seeds or [base["seed"]]
    model_name = base["model"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Grid runs live in runs/grid_<stamp>/ so compare_models.py keeps using the main runs.
    group = f"grid_{model_name}_{args.variant}_{stamp}"

    rows = []
    for lr, batch_size, weight_decay, seed in itertools.product(
        args.lrs, args.batch_sizes, args.weight_decays, seeds
    ):
        cfg_doc = copy.deepcopy(base)
        cfg_doc["variants"][args.variant]["lr"] = lr
        cfg_doc["batch_size"] = batch_size
        cfg_doc["weight_decay"] = weight_decay
        cfg_doc["seed"] = seed
        run_name = f"{group}/lr{lr:g}_bs{batch_size}_wd{weight_decay:g}_seed{seed}"
        m = run_variant(cfg_doc, args.variant, run_name=run_name)["metrics"]
        rows.append(
            {
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "seed": seed,
                "best_val_macro_f1": m["best_val_macro_f1"],
                "best_val_accuracy": m["best_val_accuracy"],
                "best_epoch": m["best_epoch"],
                "train_seconds": m["train_seconds"],
                "run_id": m["run_id"],
            }
        )

    rows.sort(key=lambda r: r["best_val_macro_f1"], reverse=True)
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = COMPARISONS_DIR / f"{group}.json"
    out_md = COMPARISONS_DIR / f"{group}.md"
    out_json.write_text(json.dumps({"base_config": str(args.config), "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        f"# Hyperparameter grid: {model_name} / {args.variant} (validation)",
        "",
        "| LR | Batch | Weight decay | Seed | Val macro-F1 | Val accuracy | Best epoch | Train s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {r['lr']:g} | {r['batch_size']} | {r['weight_decay']:g} | {r['seed']} | {r['best_val_macro_f1']:.4f} | "
            f"{r['best_val_accuracy']:.4f} | {r['best_epoch']} | {r['train_seconds']:.1f} |"
            for r in rows
        ],
        "",
        "- Sorted by val macro-F1; test set not used.",
        "",
    ]
    if len(seeds) > 1:
        lines += [
            "## Mean over seeds",
            "",
            "| LR | Batch | Weight decay | Mean val macro-F1 | Std | Runs |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        groups: dict[tuple, list[float]] = {}
        for r in rows:
            groups.setdefault((r["lr"], r["batch_size"], r["weight_decay"]), []).append(r["best_val_macro_f1"])
        for (lr, bs, wd), f1s in sorted(groups.items(), key=lambda kv: -statistics.mean(kv[1])):
            lines.append(
                f"| {lr:g} | {bs} | {wd:g} | {statistics.mean(f1s):.4f} | {statistics.stdev(f1s):.4f} | {len(f1s)} |"
            )
        lines.append("")
    md = "\n".join(lines)
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
