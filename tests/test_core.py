"""Checks for the parts that silently break experiments: split leakage and seeding."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sortwise.data.split import create_stratified_split
from sortwise.models.factory import build_model
from sortwise.training.trainer import set_seed


def _file(i: int, cls: str, sha: str | None = None) -> dict:
    return {"id": f"{cls}/{i}.jpg", "class": cls, "sha256": sha or f"{cls}-{i}", "ok": True}


def test_split_drops_duplicates_and_has_no_overlap():
    files = [_file(i, c) for c in ("glass", "metal") for i in range(20)]
    files += [
        _file(100, "glass", sha="same-a"), _file(101, "metal", sha="same-a"),  # conflicting labels
        _file(102, "glass", sha="same-b"), _file(103, "glass", sha="same-b"),  # same label
    ]
    splits = create_stratified_split({"files": files})
    all_ids = [i for ids in splits.values() for i in ids]

    assert len(all_ids) == len(set(all_ids)), "file in more than one split"
    assert "glass/100.jpg" not in all_ids and "metal/101.jpg" not in all_ids
    assert ("glass/102.jpg" in all_ids) != ("glass/103.jpg" in all_ids)
    assert len(all_ids) == 41


def test_same_seed_builds_same_head():
    set_seed(42)
    a = build_model("mobilenet_v3_small", trainable="finetune")
    set_seed(42)
    b = build_model("mobilenet_v3_small", trainable="finetune")
    assert torch.equal(a.classifier[-1].weight, b.classifier[-1].weight)


if __name__ == "__main__":
    test_split_drops_duplicates_and_has_no_overlap()
    test_same_seed_builds_same_head()
    print("OK")
