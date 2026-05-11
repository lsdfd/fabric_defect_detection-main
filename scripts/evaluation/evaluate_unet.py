from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.unet import MixedLoss, NotebookUNet, notebook_deploy_iou, segmentation_metrics
from utilities import AITEXPatchedSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a scripted U-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evaluate_split(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> dict:
    losses = []
    dice_values = []
    iou_values = []
    deploy_iou_values = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            losses.append(float(criterion(outputs, targets).item()))
            dice, iou = segmentation_metrics(outputs, targets)
            dice_values.extend(dice.detach().cpu().tolist())
            iou_values.extend(iou.detach().cpu().tolist())
            deploy_iou_values.extend(notebook_deploy_iou(outputs, targets).detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.nanmean(dice_values)) if dice_values else float("nan"),
        "jaccard": float(np.nanmean(iou_values)) if iou_values else float("nan"),
        "deploy_iou": float(np.nanmean(deploy_iou_values)) if deploy_iou_values else float("nan"),
    }


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = AITEXPatchedSegmentation(
        str(PROJECT_ROOT / "data" / "aitex"),
        transform=transforms.Compose([]),
    )
    train_samples = int(len(data) * 0.95)
    val_samples = len(data) - train_samples
    train, val = random_split(data, [train_samples, val_samples])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NotebookUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    criterion = MixedLoss(10, 2)

    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "dataset_size": len(data),
        "train_size": len(train),
        "val_size": len(val),
        "train": evaluate_split(model, train_loader, criterion, device),
        "val": evaluate_split(model, val_loader, criterion, device),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
