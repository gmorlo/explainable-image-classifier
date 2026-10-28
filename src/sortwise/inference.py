"""Load a trained run and classify one image with a Grad-CAM heatmap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from sortwise.explain.gradcam import grad_cam, overlay, preprocess
from sortwise.models.factory import MODEL_SPECS, build_model


@dataclass
class Predictor:
    model: nn.Module
    cam_layer: nn.Module
    class_names: list[str]
    device: torch.device

    def predict(self, image: Image.Image) -> tuple[dict[str, float], np.ndarray]:
        """Softmax score per class and the top-1 Grad-CAM overlay (224x224 RGB float in [0, 1])."""
        x, crop = preprocess(image)
        cam, probs = grad_cam(self.model, x.to(self.device), self.cam_layer)
        scores = {name: float(p) for name, p in zip(self.class_names, probs)}
        return scores, overlay(crop, cam)


def load_predictor(run_dir: Path, device: torch.device | None = None) -> Predictor:
    """Run dir must hold best.pt, train_config.json and label_map.json (as written by train.py)."""
    device = device or torch.device("cpu")
    model_name = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))["model_name"]
    class_names = json.loads((run_dir / "label_map.json").read_text(encoding="utf-8"))["class_names"]
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(model_name, num_classes=len(class_names), trainable=ckpt["trainable"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return Predictor(model, MODEL_SPECS[model_name].cam_layer(model), class_names, device)
