"""Upload a trained run (weights + label map + config + test metrics) to a HF Hub model repo and tag it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import RUNS_DIR

FILES = ("best.pt", "label_map.json", "train_config.json", "test_metrics.json")


def model_card(run: Path, commit: str) -> str:
    cfg = json.loads((run / "train_config.json").read_text(encoding="utf-8"))
    test = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    return (
        "---\nlibrary_name: pytorch\npipeline_tag: image-classification\n---\n\n"
        f"# SortWise — {cfg['model_name']}\n\n"
        "Waste photo classifier (6 TrashNet classes) used by the SortWise Grad-CAM demo.\n\n"
        f"- Test (TrashNet, n={test['n']}): macro-F1 **{test['macro_f1']:.3f}**, accuracy **{test['accuracy']:.3f}**\n"
        f"- Training: {cfg['trainable']}, lr {cfg['lr']}, batch {cfg['batch_size']}, seed {cfg['seed']}\n"
        f"- Code: github.com/gmorlo/explainable-image-classifier @ `{commit}`\n\n"
        "Files: `best.pt` (state dict + trainable mode), `label_map.json` (index → class), "
        "`train_config.json`, `test_metrics.json`. Load with `sortwise.inference.load_predictor`.\n\n"
        "Trained on photos of single items on a plain light background; other backgrounds are not evaluated.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUNS_DIR / "final" / "efficientnet_b0")
    parser.add_argument("--repo", default="gmorlo/sortwise-efficientnet-b0")
    parser.add_argument("--tag", required=True, help="model version, e.g. v1")
    parser.add_argument("--public", action="store_true", help="default: private repo")
    args = parser.parse_args()

    missing = [f for f in FILES if not (args.run / f).exists()]
    if missing:
        parser.error(f"{args.run} is missing {missing}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    if args.tag in {t.name for t in api.list_repo_refs(args.repo).tags}:
        parser.error(f"tag {args.tag} already exists in {args.repo}; versions are immutable, pick a new one")
    info = api.upload_folder(repo_id=args.repo, folder_path=args.run, allow_patterns=list(FILES),
                             commit_message=f"Model {args.tag} from {commit}")
    api.upload_file(repo_id=args.repo, path_or_fileobj=model_card(args.run, commit).encode(),
                    path_in_repo="README.md", commit_message=f"Model card {args.tag}")
    api.create_tag(args.repo, tag=args.tag)
    print(f"Published {args.repo}@{args.tag} ({info.oid[:7]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
