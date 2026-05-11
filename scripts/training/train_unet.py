from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.unet import MixedLoss, NotebookUNet, notebook_deploy_iou, segmentation_metrics
from utilities import AITEXPatchedSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted reproduction of train/unet_segmentation.ipynb.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "unet_repro")
    parser.add_argument("--eval-each-epoch", action="store_true")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> dict:
    model.eval()
    losses = []
    dice_values = []
    jaccard_values = []
    deploy_iou_values = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            losses.append(float(criterion(outputs, targets).item()))
            dice, jaccard = segmentation_metrics(outputs, targets)
            dice_values.extend(dice.detach().cpu().tolist())
            jaccard_values.extend(jaccard.detach().cpu().tolist())
            deploy_iou_values.extend(notebook_deploy_iou(outputs, targets).detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.nanmean(dice_values)) if dice_values else float("nan"),
        "jaccard": float(np.nanmean(jaccard_values)) if jaccard_values else float("nan"),
        "deploy_iou": float(np.nanmean(deploy_iou_values)) if deploy_iou_values else float("nan"),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NotebookUNet().to(device)
    criterion = MixedLoss(10, 2)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_started_at = time.time()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_seconds": time.time() - epoch_started_at,
        }
        if args.eval_each_epoch or epoch == args.epochs:
            row["train_eval"] = evaluate(model, train_loader, criterion, device)
            row["val_eval"] = evaluate(model, val_loader, criterion, device)
        history.append(row)

        eval_text = ""
        if "val_eval" in row:
            eval_text = (
                f", Val Loss: {row['val_eval']['loss']:.5f}, "
                f"Val Dice: {row['val_eval']['dice']:.5f}, "
                f"Val IoU: {row['val_eval']['jaccard']:.5f}, "
                f"Val Deploy IoU: {row['val_eval']['deploy_iou']:.5f}"
            )
        print(
            f"Epoch: {epoch}/{args.epochs}, "
            f"Train Loss: {row['train_loss']:.5f}, "
            f"Epoch Seconds: {row['epoch_seconds']:.2f}"
            f"{eval_text}",
            flush=True,
        )

    checkpoint = args.output_dir / "unet_repro_last.pt"
    torch.save(model.state_dict(), checkpoint)

    if "train_eval" not in history[-1]:
        history[-1]["train_eval"] = evaluate(model, train_loader, criterion, device)
        history[-1]["val_eval"] = evaluate(model, val_loader, criterion, device)

    result = {
        "history": history,
        "final": history[-1],
        "checkpoint": str(checkpoint),
        "device": str(device),
        "dataset_size": len(data),
        "train_size": len(train),
        "val_size": len(val),
        "total_seconds": time.time() - started_at,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
