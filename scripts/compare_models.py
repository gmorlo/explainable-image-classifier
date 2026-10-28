"""Compare latest run of every model x variant on validation metrics and inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import COMPARISONS_DIR, RUNS_DIR
from sortwise.inference import load_predictor


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_runs() -> dict[tuple[str, str], Path]:
    """Newest completed run directory per (model_name, variant)."""
    latest: dict[tuple[str, str], Path] = {}
    for run_dir in RUNS_DIR.iterdir():
        if not (run_dir / "metrics.json").is_file():
            continue
        metrics = _load_json(run_dir / "metrics.json")
        # Runs before multi-model support store model_name only in train_config.json.
        model_name = metrics.get("model_name") or _load_json(run_dir / "train_config.json")["model_name"]
        key = (model_name, metrics["variant"])
        if key not in latest or run_dir.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = run_dir
    return latest


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, device: torch.device, *, warmup: int = 5, repeats: int = 30) -> float:
    """Median forward time for one 224x224 image."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    times = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-latency", action="store_true", help="Do not measure inference time")
    args = parser.parse_args()

    rows = []
    for (model_name, variant), run_dir in sorted(latest_runs().items()):
        metrics = _load_json(run_dir / "metrics.json")
        row = {
            "model": model_name,
            "variant": variant,
            "run_id": metrics["run_id"],
            "best_val_macro_f1": metrics["best_val_macro_f1"],
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_epoch": metrics["best_epoch"],
            "train_seconds": metrics["train_seconds"],
            "trainable_parameters": metrics["trainable_parameters"],
            "weights_mb": metrics["weights_bytes"] / 1e6,
            "gpu_name": metrics.get("gpu_name"),
        }
        if not args.skip_latency:
            model = load_predictor(run_dir).model
            row["cpu_ms"] = measure_latency_ms(model, torch.device("cpu"))
            if torch.cuda.is_available():
                row["gpu_ms"] = measure_latency_ms(model, torch.device("cuda"))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed runs under {RUNS_DIR}")
    winner = max(rows, key=lambda r: r["best_val_macro_f1"])

    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = COMPARISONS_DIR / "model_comparison.json"
    out_md = COMPARISONS_DIR / "model_comparison.md"
    out_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metric": "best_val_macro_f1",
                "winner": {"model": winner["model"], "variant": winner["variant"]},
                "rows": rows,
                "notes": [
                    "Validation metrics only; test set was not used.",
                    "Latency: median of 30 forward passes, batch 1, 224x224, after 5 warmup passes.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fmt_ms(row: dict, key: str) -> str:
        return f"{row[key]:.1f}" if key in row else "-"

    lines = [
        "# Model comparison (validation)",
        "",
        f"Best by val macro-F1: **{winner['model']} / {winner['variant']}**",
        "",
        "| Model | Variant | Val macro-F1 | Val accuracy | Best epoch | Train s | Trainable params | Weights MB | CPU ms | GPU ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['best_val_macro_f1']:.4f} | {r['best_val_accuracy']:.4f} | "
            f"{r['best_epoch']} | {r['train_seconds']:.1f} | {r['trainable_parameters']:,} | "
            f"{r['weights_mb']:.1f} | {fmt_ms(r, 'cpu_ms')} | {fmt_ms(r, 'gpu_ms')} |"
        )
    lines += [
        "",
        "- Test set not used for this comparison.",
        "- Latency: median of 30 forward passes, batch 1, 224x224.",
        "",
        "Runs:",
        *[f"- {r['model']} / {r['variant']}: `{r['run_id']}`" for r in rows],
        "",
    ]
    md = "\n".join(lines)
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
