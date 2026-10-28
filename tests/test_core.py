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


def test_grad_cam_works_for_frozen_and_finetuned_models():
    from sortwise.explain.gradcam import grad_cam
    from sortwise.models.factory import MODEL_SPECS

    x = torch.randn(1, 3, 224, 224)
    for name in MODEL_SPECS:
        for trainable in ("head", "finetune"):
            model = build_model(name, trainable=trainable)
            cam, probs = grad_cam(model, x, MODEL_SPECS[name].cam_layer(model))
            assert cam.shape == (224, 224) and 0.0 <= cam.min() and cam.max() <= 1.0, (name, trainable)
            assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_predictor_returns_all_classes_and_heatmap():
    import numpy as np
    from PIL import Image

    from sortwise.config import CLASS_LABELS_PL, CLASS_NAMES
    from sortwise.inference import Predictor
    from sortwise.models.factory import MODEL_SPECS

    model = build_model("mobilenet_v3_small", trainable="finetune")
    predictor = Predictor(model, MODEL_SPECS["mobilenet_v3_small"].cam_layer(model), list(CLASS_NAMES),
                          torch.device("cpu"))
    image = Image.fromarray(np.random.randint(0, 255, (300, 400, 4), dtype=np.uint8))  # RGBA, non-square
    scores, heatmap = predictor.predict(image)
    assert list(scores) == list(CLASS_NAMES) and set(CLASS_LABELS_PL) == set(CLASS_NAMES)
    assert abs(sum(scores.values()) - 1.0) < 1e-4
    assert heatmap.shape == (224, 224, 3) and 0.0 <= heatmap.min() and heatmap.max() <= 1.0


if __name__ == "__main__":
    test_split_drops_duplicates_and_has_no_overlap()
    test_same_seed_builds_same_head()
    test_grad_cam_works_for_frozen_and_finetuned_models()
    test_predictor_returns_all_classes_and_heatmap()
    print("OK")
