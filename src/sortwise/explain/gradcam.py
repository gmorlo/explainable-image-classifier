"""Grad-CAM: which image regions pushed the model towards a class."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from sortwise.data.transforms import build_eval_transforms


def preprocess(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Model input (1, 3, 224, 224) and the same 224x224 crop as RGB uint8 for display."""
    tf = build_eval_transforms()
    crop = transforms.Compose(tf.transforms[:2])(image.convert("RGB"))  # Resize + CenterCrop
    return tf(image.convert("RGB")).unsqueeze(0), np.asarray(crop)


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Return (heatmap HxW in [0, 1], softmax scores) for one image.

    class_idx=None explains the predicted class.
    """
    model.eval()
    store: dict[str, torch.Tensor] = {}

    def keep(_module, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = target_layer.register_forward_hook(keep)
    try:
        # enable_grad: callers (e.g. an app) may run inference under torch.no_grad().
        with torch.enable_grad():
            # Input grad keeps the graph alive when the backbone is frozen (head-only models).
            x = x.clone().requires_grad_(True)
            logits = model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.cpu().numpy(), logits.softmax(dim=1)[0].detach().cpu()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap onto an RGB uint8 image."""
    heat = colormaps["jet"](cam)[..., :3]
    return ((1 - alpha) * image / 255.0 + alpha * heat).clip(0, 1)
