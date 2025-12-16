# models.py
import torch.nn as nn
from torchvision import models as tv_models


class TinyCNN(nn.Module):
    """
    Very small CNN for low-spec devices.
    ~100k parameters (depends slightly on num_classes).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SmallCNN(nn.Module):
    """
    Slightly larger CNN than TinyCNN, but still light.
    ~300k-500k parameters depending on classes.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def _make_resnet18(num_classes: int, pretrained: bool):
    # New torchvision uses "weights" instead of pretrained=True
    if pretrained:
        weights = tv_models.ResNet18_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.resnet18(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


def _make_mobilenet_v2(num_classes: int, pretrained: bool):
    if pretrained:
        weights = tv_models.MobileNet_V2_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.mobilenet_v2(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def _make_efficientnet_b0(num_classes: int, pretrained: bool):
    if pretrained:
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None
    net = tv_models.efficientnet_b0(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def get_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Factory function to get a model by name.

    Supported names:
      - "tiny_cnn"      (custom small CNN)
      - "small_cnn"     (custom slightly larger CNN)
      - "resnet18"
      - "mobilenet_v2"
      - "efficientnet_b0"
    """
    name = name.lower()

    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    elif name == "small_cnn":
        return SmallCNN(num_classes)
    elif name == "resnet18":
        return _make_resnet18(num_classes, pretrained=pretrained)
    elif name == "mobilenet_v2":
        return _make_mobilenet_v2(num_classes, pretrained=pretrained)
    elif name == "efficientnet_b0":
        return _make_efficientnet_b0(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")
