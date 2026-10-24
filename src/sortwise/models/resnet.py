"""ResNet18 builders for head-only and layer4 fine-tuning."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_resnet18(num_classes: int = 6, *, trainable: str = "head") -> nn.Module:
    """
    Build ImageNet-pretrained ResNet18 with a 6-class head.

    trainable:
      - "head": freeze backbone, train fc only
      - "layer4_fc": train layer4 + fc
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if trainable == "head":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    elif trainable == "layer4_fc":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    else:
        raise ValueError(f"Unknown trainable mode: {trainable!r}")

    return model
