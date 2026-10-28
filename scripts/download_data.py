"""Download TrashNet dataset-resized and build a checksummed manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sortwise.data.download import download_trashnet
from sortwise.data.manifest import build_manifest, summarize_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if dataset already present",
    )
    args = parser.parse_args()

    dataset_dir = download_trashnet(force=args.force)
    print(f"Dataset: {dataset_dir}")

    manifest = build_manifest(dataset_dir)
    print(summarize_manifest(manifest))
    print("Manifest written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
