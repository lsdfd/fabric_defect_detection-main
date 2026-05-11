"""Fast script reproduction of train/binary_patch_classification.ipynb.

This keeps the original binary classifier, split, weighted sampler, BCELoss,
and Adam settings, but caches resized patch tensors and evaluates in batches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(TRAIN_DIR))

from model_architectures import BinaryClassifier
from utilities import AITEXPatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cached binary classifier training.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--antialias", choices=["false", "true"], default="false")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "binary_cached_30ep")
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--eval-full-each-epoch", action="store_true")
    return parser.parse_args()


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (outputs >= 0.5).float()
    return (predictions == labels).sum().item() / labels.size(0)


def build_cached_dataset(aitex_dir: Path, antialias: bool) -> tuple[TensorDataset, list[int]]:
    # Identity transform here so AITEXPatched performs the original patching and
    # histogram equalization, while resize is cached exactly once below.
    source = AITEXPatched(str(aitex_dir), transform=lambda x: x, greyscale=True)
    images = torch.stack(
        [TF.resize(img, (224, 224), antialias=antialias) for img in source.patched_images]
    ).float()
    labels = torch.tensor(source.has_defect, dtype=torch.long)
    return TensorDataset(images, labels), source.has_defect


def evaluate_f1(model: nn.Module, dataset: TensorDataset, device: torch.device, batch_size: int, num_workers: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend((outputs.detach().cpu().view(-1) >= 0.5).long().tolist())
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_started_at = time.time()
    dataset, labels_for_weights = build_cached_dataset(
        PROJECT_ROOT / "data" / "aitex",
        antialias=args.antialias == "true",
    )
    dataset_seconds = time.time() - dataset_started_at

    num_samples = len(dataset)
    train_samples = int(num_samples * 0.95)
    val_samples = num_samples - train_samples
    train, val = random_split(dataset, [train_samples, val_samples])

    class_counts = [labels_for_weights.count(c) for c in range(2)]
    total_samples = sum(class_counts)
    class_weights_list = [total_samples / (2 * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights_list).to(device)

    sample_weights = []
    for _img, label in train:
        sample_weights.append(class_weights[int(label)].detach().cpu())

    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = BinaryClassifier().to(device)
    loss_fn = nn.BCELoss(weight=class_weights[-1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    total_started_at = time.time()
    for epoch in range(args.epochs):
        epoch_started_at = time.time()
        train_loss = 0.0
        train_accuracy = 0.0
        valid_loss = 0.0
        valid_accuracy = 0.0

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
            row.update({f"full_{k}": v for k, v in evaluate_f1(model, dataset, device, args.eval_batch_size, args.num_workers).items()})
        history.append(row)

        if args.save_each_epoch:
            torch.save(model.state_dict(), args.output_dir / f"binary_cached_epoch_{epoch + 1:03d}.pt")

        full_text = f", Full F1: {row['full_f1']:.4f}" if "full_f1" in row else ""
        print(
            f"Epoch: {epoch + 1}/{args.epochs}, "
            f"Train Loss: {row['train_loss']:.4f}, "
            f"Train Accuracy: {row['train_accuracy']:.4f}, "
            f"Valid Loss: {row['valid_loss']:.4f}, "
            f"Valid Accuracy: {row['valid_accuracy']:.4f}, "
            f"Epoch Seconds: {row['epoch_seconds']:.2f}"
            f"{full_text}",
            flush=True,
        )

    checkpoint = args.output_dir / "binary_cached_last.pt"
    torch.save(model.state_dict(), checkpoint)
    full_eval = evaluate_f1(model, dataset, device, args.eval_batch_size, args.num_workers)

    result = {
        "history": history,
        "full_dataset_f1": full_eval["f1"],
        "confusion_matrix": full_eval["confusion_matrix"],
        "class_counts": class_counts,
        "dataset_seconds": dataset_seconds,
        "total_seconds": time.time() - total_started_at,
        "device": str(device),
        "checkpoint": str(checkpoint),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("F1 Score:", full_eval["f1"], flush=True)
    print("Confusion Matrix:", full_eval["confusion_matrix"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
