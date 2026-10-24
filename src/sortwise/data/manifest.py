"""Build and load a file-level dataset manifest with SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from sortwise.config import (
    CLASS_NAMES,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    MANIFEST_PATH,
    PROCESSED_DIR,
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _probe_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return {
            "ok": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — record any decode failure
        return {
            "ok": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": str(exc),
        }


def build_manifest(
    dataset_dir: Path | None = None,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Scan class folders, checksum files, probe images, write manifest JSON."""
    root = dataset_dir or DATASET_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            probe = _probe_image(path)
            files.append(
                {
                    "id": rel,
                    "path": rel,
                    "class": class_name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    **probe,
                }
            )

    counts = Counter(f["class"] for f in files)
    corrupt = [f["id"] for f in files if not f["ok"]]

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "num_files": len(files),
        "class_counts": {name: counts.get(name, 0) for name in CLASS_NAMES},
        "corrupt_files": corrupt,
        "files": files,
    }

    out = output_path or MANIFEST_PATH
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        f"Files: {manifest['num_files']}",
        f"Corrupt: {len(manifest.get('corrupt_files', []))}",
        "Class counts:",
    ]
    for name, count in manifest.get("class_counts", {}).items():
        lines.append(f"  {name}: {count}")
    return "\n".join(lines)
