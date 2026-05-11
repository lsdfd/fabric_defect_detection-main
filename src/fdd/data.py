from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torch.utils.data.sampler import WeightedRandomSampler


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    val: Dataset


class AITEXPatchDataset(Dataset):
    """AITEX fabric patches for binary defect classification.

    This mirrors the original notebook pipeline while fixing platform-specific
    path parsing. Full images are resized to 256 x 4096, histogram-equalized,
    and split into sixteen 256 x 256 grayscale patches.
    """

    image_dims = (256, 4096)
    patch_size = 256

    def __init__(self, aitex_dir: str | Path, transform=None, greyscale: bool = True):
        self.aitex_dir = Path(aitex_dir)
        self.transform = transform
        self.greyscale = greyscale

        normal_dir = self.aitex_dir / "NODefect_images"
        defect_dir = self.aitex_dir / "Defect_images"
        mask_dir = self.aitex_dir / "Mask_images"

        self.normal_images = sorted(normal_dir.rglob("*.png"))
        self.defect_masks = sorted(mask_dir.rglob("*_mask.png"))
        self.mask_roots = [mask.name.removesuffix("_mask.png") for mask in self.defect_masks]
        self.defect_images = [defect_dir / f"{root}.png" for root in self.mask_roots]

        missing = [str(path) for path in self.defect_images if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing defect images for masks: {missing[:5]}")

        self.image_paths = [*self.normal_images, *self.defect_images]
        read_mode = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR
        self.images = [cv2.imread(str(path), read_mode) for path in self.image_paths]
        if any(image is None for image in self.images):
            bad = [str(path) for path, image in zip(self.image_paths, self.images) if image is None]
            raise ValueError(f"Could not read images: {bad[:5]}")

        normal_masks = [np.zeros(self.image_dims, dtype=np.uint8) for _ in self.normal_images]
        defect_masks = [
            cv2.threshold(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), 0, 1, cv2.THRESH_BINARY)[1]
            for path in self.defect_masks
        ]
        self.masks = [*normal_masks, *defect_masks]

        self.patches: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.source_paths: list[str] = []
        self.patch_indices: list[int] = []

        for image_path, image, mask in zip(self.image_paths, self.images, self.masks):
            image_resized = cv2.resize(image, (4096, 256))
            image_resized = cv2.equalizeHist(image_resized) / 255.0
            mask_resized = cv2.resize(mask, (4096, 256))

            for patch_idx, start in enumerate(range(0, 4096, self.patch_size)):
                image_patch = image_resized[:, start : start + self.patch_size]
                mask_patch = mask_resized[:, start : start + self.patch_size]
                self.patches.append(torch.tensor(image_patch, dtype=torch.float32).reshape(1, 256, 256))
                self.labels.append(int(mask_patch.sum() > 0))
                self.source_paths.append(str(image_path))
                self.patch_indices.append(patch_idx)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        image = self.patches[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)


def make_splits(dataset: Dataset, train_fraction: float = 0.95, seed: Optional[int] = 42) -> DatasetSplits:
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [train_size, val_size], generator=generator)
    return DatasetSplits(train=train, val=val)


def make_balanced_sampler(subset: Subset, labels: list[int]) -> WeightedRandomSampler:
    subset_labels = torch.tensor([labels[idx] for idx in subset.indices], dtype=torch.long)
    class_counts = torch.bincount(subset_labels, minlength=2).float()
    class_weights = 1.0 / class_counts.clamp_min(1)
    sample_weights = class_weights[subset_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def resolve_aitex_dir(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "aitex"
