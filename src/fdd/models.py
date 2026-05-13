from __future__ import annotations

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """Original fabric teacher architecture from the upstream project."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class OpticalStudentClassifier(nn.Module):
    """One optical-style convolution followed by two fully-connected layers.

    This mirrors the compressed student in the reference optical encoder paper:
    a single convolutional frontend whose kernels are future PSF targets, followed
    by a lightweight digital readout head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        optical_kernels: int = 16,
        kernel_size: int = 7,
        pooled_size: int = 6,
        hidden_dim: int = 256,
        optical_activation: str = "relu",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so same-padding is symmetric.")
        if optical_activation not in {"relu", "identity"}:
            raise ValueError("optical_activation must be 'relu' or 'identity'.")

        self.optical_activation = optical_activation
        self.optical = nn.Conv2d(
            in_channels,
            optical_kernels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.backend = nn.Sequential(
            nn.Flatten(),
            nn.Linear(optical_kernels * pooled_size * pooled_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.optical(x)
        if self.optical_activation == "relu":
            x = torch.relu(x)
        x = self.pool(x)
        return self.backend(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def optical_kernels(self) -> torch.Tensor:
        return self.optical.weight.detach().cpu()


class OpticalSegmentationStudent(nn.Module):
    """Optical-style segmentation student with a minimal electronic readout head."""

    def __init__(
        self,
        in_channels: int = 1,
        optical_kernels: int = 32,
        kernel_size: int = 5,
        backend_channels: int = 16,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so same-padding is symmetric.")

        self.optical = nn.Conv2d(
            in_channels,
            optical_kernels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.backend = nn.Sequential(
            nn.Conv2d(optical_kernels, backend_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(backend_channels, backend_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(backend_channels, 1, kernel_size=1, bias=True),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend(self.optical(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def optical_kernels(self) -> torch.Tensor:
        return self.optical.weight.detach().cpu()


def load_unet_teacher(checkpoint_path: str, device: torch.device) -> nn.Module:
    from fdd.unet import NotebookUNet

    model = NotebookUNet().to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def load_teacher(checkpoint_path: str, device: torch.device) -> BinaryClassifier:
    model = BinaryClassifier().to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model
