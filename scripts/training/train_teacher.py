from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXPatchDataset, make_balanced_sampler, make_splits, resolve_aitex_dir
from fdd.models import BinaryClassifier
from fdd.training import binary_metrics_from_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the original fabric BinaryClassifier teacher.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None, help="Default None matches the original notebook random split.")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-positive-loss-weight", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            probs = model(images)
            loss = loss_fn(probs, labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    return binary_metrics_from_probs(
        torch.cat(all_probs).view(-1),
        torch.cat(all_labels).view(-1),
        torch.tensor(total_loss / max(total_count, 1)),
    )


def class_weights_from_labels(labels: list[int], device: torch.device) -> torch.Tensor:
    class_counts = torch.tensor([labels.count(c) for c in range(2)], dtype=torch.float32, device=device)
    total_samples = class_counts.sum()
    return total_samples / (2 * class_counts.clamp_min(1))


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or args.project_root / "outputs" / "teacher"
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((224, 224))])
    dataset = AITEXPatchDataset(resolve_aitex_dir(args.project_root), transform=transform)
    splits = make_splits(dataset, seed=args.seed)
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(splits.train, dataset.labels)
    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)

    model = BinaryClassifier().to(device)
    class_weights = class_weights_from_labels(dataset.labels, device=device)
    loss_weight = None if args.no_positive_loss_weight else class_weights[-1]
    loss_fn = torch.nn.BCELoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_f1 = -1.0
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        epoch_started_at = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            probs = model(images)
            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

        val_metrics = evaluate(model, val_loader, device, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val": val_metrics.__dict__,
            "epoch_seconds": time.time() - epoch_started_at,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            torch.save(model.state_dict(), output_dir / "binary_classifier_best.pt")

    torch.save(model.state_dict(), output_dir / "binary_classifier_last.pt")
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    full_metrics = evaluate(model, full_loader, device, loss_fn)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    (output_dir / "full_dataset_metrics.json").write_text(json.dumps(full_metrics.__dict__, indent=2, ensure_ascii=False))
    print(f"full_dataset_metrics={json.dumps(full_metrics.__dict__, ensure_ascii=False)}")
    print(f"finished in {time.time() - started_at:.1f}s; best_val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
