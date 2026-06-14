from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms


IMAGE_SIZE = 224


def build_transforms(image_size: int = IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/eval transforms.

    Train transform includes augmentation. Eval transform is deterministic.
    ImageNet mean/std are used because the selected models are ImageNet-pretrained.
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def _set_parameter_requires_grad(model: nn.Module, feature_extract: bool) -> None:
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def create_model(model_name: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True, feature_extract: bool = False) -> nn.Module:
    """Create a binary classifier.

    Available models:
    - efficientnet_b0: default. Strong accuracy/speed balance.
    - resnet18: stable baseline. Good for comparison.
    - mobilenet_v3_small: lightweight model. Good for fast inference.
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        _set_parameter_requires_grad(model, feature_extract)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        _set_parameter_requires_grad(model, feature_extract)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        _set_parameter_requires_grad(model, feature_extract)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_name: {model_name}")


MODEL_PRESETS = {
    "efficientnet_b0": {
        "lr": 3e-4,
        "batch_size": 32,
        "epochs": 15,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
    },
    "resnet18": {
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 12,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
    },
    "mobilenet_v3_small": {
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 15,
        "weight_decay": 5e-5,
        "optimizer": "adamw",
    },
}
