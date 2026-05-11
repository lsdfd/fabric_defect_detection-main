from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXPatchDataset, make_splits, resolve_aitex_dir
from fdd.models import load_teacher
from fdd.training import binary_metrics_from_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the original fabric binary teacher.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint
    if checkpoint is None:
        trained_teacher = args.project_root / "outputs" / "teacher" / "binary_classifier_best.pt"
        checkpoint = trained_teacher if trained_teacher.exists() else args.project_root / "models" / "bigger_binary_F1_0.98.pth"

    transform = transforms.Compose([transforms.Resize((224, 224))])
    dataset = AITEXPatchDataset(resolve_aitex_dir(args.project_root), transform=transform)
    splits = make_splits(dataset, seed=args.seed)
    loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)

    model = load_teacher(str(checkpoint), device=device)
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            probs = model(images)
            loss = torch.nn.functional.binary_cross_entropy(probs, labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

    probs = torch.cat(all_probs).view(-1)
    labels = torch.cat(all_labels).view(-1)
    mean_loss = torch.tensor(total_loss / max(total_count, 1))
    metrics = binary_metrics_from_probs(probs, labels, mean_loss)
    print(metrics)


if __name__ == "__main__":
    main()
