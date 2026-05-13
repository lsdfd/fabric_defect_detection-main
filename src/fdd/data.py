from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
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

        self.normal_images = sorted(path for path in normal_dir.rglob("*.png") if not self._is_artifact(path))
        self.defect_masks = sorted(path for path in mask_dir.rglob("*_mask.png") if not self._is_artifact(path))
        self.mask_roots = [mask.name.removesuffix("_mask.png") for mask in self.defect_masks]
        self.defect_images = [defect_dir / f"{root}.png" for root in self.mask_roots]

        missing = [str(path) for path in self.defect_images if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing defect images for masks: {missing[:5]}")

        self.image_paths = [*self.normal_images, *self.defect_images]
        normal_masks = [np.zeros(self.image_dims, dtype=np.uint8) for _ in self.normal_images]
        defect_masks = [self._load_mask(path) for path in self.defect_masks]
        self.masks = [*normal_masks, *defect_masks]

        self.patches: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.source_paths: list[str] = []
        self.patch_indices: list[int] = []

        for image_path, mask in zip(self.image_paths, self.masks):
            image_resized = self._load_image(image_path)
            mask_resized = self._resize_mask(mask)

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

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("L") if self.greyscale else image.convert("RGB")
            image = image.resize((4096, 256), resample=Image.Resampling.BILINEAR)
            if self.greyscale:
                image = ImageOps.equalize(image)
                return np.asarray(image, dtype=np.float32) / 255.0

            return np.asarray(image, dtype=np.float32) / 255.0

    def _load_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as mask:
            mask = mask.convert("L")
            mask_array = np.asarray(mask, dtype=np.uint8)
        return (mask_array > 0).astype(np.uint8)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        pil_mask = Image.fromarray(mask * 255)
        resized = pil_mask.resize((4096, 256), resample=Image.Resampling.NEAREST)
        return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)

    @staticmethod
    def _is_artifact(path: Path) -> bool:
        return path.name.startswith("._")


class AITEXSegmentationPatchDataset(Dataset):
    """AITEX defect-only patches for binary segmentation."""

    image_dims = (256, 4096)
    patch_size = 256

    def __init__(self, aitex_dir: str | Path, transform=None):
        self.aitex_dir = Path(aitex_dir)
        self.transform = transform
        self.classification_view = AITEXPatchDataset(aitex_dir, transform=None, greyscale=True)

        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.source_paths: list[str] = []
        self.patch_indices: list[int] = []

        for image_path, mask in zip(
            self.classification_view.image_paths,
            self.classification_view.masks,
        ):
            image_resized = self.classification_view._load_image(Path(image_path))
            mask_resized = self.classification_view._resize_mask(mask)
            for patch_idx, start in enumerate(range(0, 4096, self.patch_size)):
                image_patch = image_resized[:, start : start + self.patch_size]
                mask_patch = mask_resized[:, start : start + self.patch_size]
                if mask_patch.sum() == 0:
                    continue
                self.images.append(torch.tensor(image_patch, dtype=torch.float32).reshape(1, 256, 256))
                self.masks.append(torch.tensor(mask_patch, dtype=torch.float32).reshape(1, 256, 256))
                self.source_paths.append(str(image_path))
                self.patch_indices.append(patch_idx)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


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
