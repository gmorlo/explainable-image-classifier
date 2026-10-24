"""Create a stratified 70/15/15 train/val/test split from the manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.config import SPLIT_RATIOS, SPLIT_SEED
from sortwise.data.manifest import load_manifest
from sortwise.data.split import create_stratified_split, save_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    manifest = load_manifest()
    splits = create_stratified_split(manifest, seed=args.seed)
    out = save_splits(splits, manifest, seed=args.seed)

    print(f"Seed: {args.seed}")
    print(f"Ratios: {SPLIT_RATIOS}")
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
