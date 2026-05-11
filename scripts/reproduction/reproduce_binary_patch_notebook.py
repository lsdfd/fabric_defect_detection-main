"""Script version of train/binary_patch_classification.ipynb.

This intentionally follows the notebook step by step. It only changes what is
needed to run outside Jupyter and on non-CUDA machines:

- adds train/ to sys.path so notebook imports work;
- uses the original AITEXPatched and BinaryClassifier;
- keeps Resize, random_split, WeightedRandomSampler, BCELoss weight, Adam, 5 epochs;
- saves under outputs/notebook_repro instead of overwriting the Git LFS pointer in models/;
- evaluates F1 on the full dataset just like the notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(TRAIN_DIR))

from model_architectures import BinaryClassifier
from utilities import AITEXPatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce binary_patch_classification.ipynb faithfully.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "notebook_repro")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None, help="Notebook did not set a seed; default keeps that behavior.")
    parser.add_argument("--antialias", choices=["current", "false", "true"], default="false")
    parser.add_argument("--eval-full-each-epoch", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-each-epoch", action="store_true")
    return parser.parse_args()


def calculate_accuracy(outputs, labels):
    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def resize_transform(antialias: str):
    if antialias == "current":
        return transforms.Compose([transforms.Resize((224, 224))])
    return transforms.Compose([transforms.Resize((224, 224), antialias=(antialias == "true"))])


def full_dataset_f1(model, data, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img, label in data:
            res = model(img.reshape((1, 1, 224, 224)).to(device))
            y_true.append(label)
            y_pred.append(int(res.cpu().detach() >= 0.5))
    return f1_score(y_true, y_pred)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    root = PROJECT_ROOT
    data_dir = root / "data"
    aitex_dir = data_dir / "aitex"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = resize_transform(args.antialias)
    data = AITEXPatched(str(aitex_dir), transform=transform, greyscale=True)
    num_samples = len(data)
    train_samples = int(num_samples * 0.95)
    val_samples = num_samples - train_samples
    train, val = random_split(data, [train_samples, val_samples])

    class_counts = [data.has_defect.count(c) for c in range(2)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (2 * count) for count in class_counts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.FloatTensor(class_weights).to(device)

    sample_weights = [0] * len(train)
    for idx, (_img, label) in enumerate(train):
        sample_weights[idx] = class_weights[int(label)].detach().cpu()

    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BinaryClassifier().to(device)
    loss_fn = nn.BCELoss(weight=class_weights[-1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_full_f1 = -1.0
    best_checkpoint = args.output_dir / "bigger_binary_F1_repro_best.pt"
    started_at = time.time()
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        valid_loss = 0.0
        valid_accuracy = 0.0
        epoch_started_at = time.time()

        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_accuracy += calculate_accuracy(outputs, labels)

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                valid_accuracy += calculate_accuracy(outputs, labels)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train),
            "train_accuracy": train_accuracy / len(train_loader),
            "valid_loss": valid_loss / len(val),
            "valid_accuracy": valid_accuracy / len(val_loader),
            "epoch_seconds": time.time() - epoch_started_at,
        }
        if args.eval_full_each_epoch:
            row["full_dataset_f1"] = full_dataset_f1(model, data, device)
            if row["full_dataset_f1"] > best_full_f1:
                best_full_f1 = row["full_dataset_f1"]
                torch.save(model.state_dict(), best_checkpoint)
        history.append(row)
        if args.save_each_epoch:
            epoch_checkpoint = args.output_dir / f"bigger_binary_epoch_{epoch + 1:03d}.pt"
            torch.save(model.state_dict(), epoch_checkpoint)
        full_text = f", Full F1: {row['full_dataset_f1']:.4f}" if "full_dataset_f1" in row else ""
        print(
            f"Epoch: {epoch+1}/{args.epochs}, "
            f"Train Loss: {row['train_loss']:.4f}, "
            f"Train Accuracy: {row['train_accuracy']:.4f}, "
            f"Valid Loss: {row['valid_loss']:.4f}, "
            f"Valid Accuracy: {row['valid_accuracy']:.4f}"
            f"{full_text}",
            flush=True,
        )

    checkpoint = args.output_dir / "bigger_binary_F1_repro.pt"
    torch.save(model.state_dict(), checkpoint)

    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()

    full_f1 = full_dataset_f1(model, data, device)
    print("F1 Score: ", full_f1, flush=True)

    result = {
        "history": history,
        "full_dataset_f1": full_f1,
        "class_counts": class_counts,
        "total_seconds": time.time() - started_at,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "antialias": args.antialias,
    }
    (args.output_dir / "repro_history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
