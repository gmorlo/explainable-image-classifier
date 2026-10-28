"""ImageNet-pretrained backbones with a 6-class head and head/finetune freezing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[], nn.Module]
    replace_head: Callable[[nn.Module, int], None]
    head_prefixes: tuple[str, ...]
    finetune_prefixes: tuple[str, ...]  # last stage, trained together with the head


def _resnet18_head(model: nn.Module, num_classes: int) -> None:
    model.fc = nn.Linear(model.fc.in_features, num_classes)


def _last_linear_head(model: nn.Module, num_classes: int) -> None:
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        build=lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        replace_head=_resnet18_head,
        head_prefixes=("fc.",),
        finetune_prefixes=("layer4.",),
    ),
    "efficientnet_b0": ModelSpec(
        build=lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        replace_head=_last_linear_head,
        head_prefixes=("classifier.",),
        finetune_prefixes=("features.6.", "features.7.", "features.8."),
    ),
    "mobilenet_v3_small": ModelSpec(
        build=lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        replace_head=_last_linear_head,
        head_prefixes=("classifier.",),
        finetune_prefixes=("features.9.", "features.10.", "features.11.", "features.12."),
    ),
}

# "layer4_fc" is the original name of ResNet18 finetune mode (older configs and runs).
TRAINABLE_ALIASES = {"layer4_fc": "finetune"}


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, num_classes: int = 6, *, trainable: str = "head") -> nn.Module:
    """
    Build a pretrained backbone with a fresh head.

    trainable:
      - "head": freeze backbone, train classifier only
      - "finetune": train last stage + classifier
    """
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {list(MODEL_SPECS)}")
    trainable = TRAINABLE_ALIASES.get(trainable, trainable)
    spec = MODEL_SPECS[name]
    if trainable == "head":
        prefixes = spec.head_prefixes
    elif trainable == "finetune":
        prefixes = spec.head_prefixes + spec.finetune_prefixes
    else:
        raise ValueError(f"Unknown trainable mode: {trainable!r}")

    model = spec.build()
    spec.replace_head(model, num_classes)
    for param_name, param in model.named_parameters():
        param.requires_grad = param_name.startswith(prefixes)
    return model
