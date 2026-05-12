from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXPatchDataset, make_splits, resolve_aitex_dir
from fdd.models import OpticalStudentClassifier
from fdd.training import binary_metrics_from_probs, threshold_sweep_from_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained optical student.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optical-kernels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooled-size", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--optical-activation", type=str, default="relu", choices=["relu", "identity"])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_size: int,
    threshold: float,
    threshold_sweep: bool,
):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            if input_size != 256:
                images = torch.nn.functional.interpolate(
                    images,
                    size=(input_size, input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            logits = model.forward_logits(images)
            probs = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    probs = torch.cat(all_probs).view(-1)
    labels = torch.cat(all_labels).view(-1)
    loss = torch.tensor(total_loss / max(total_count, 1))
    if threshold_sweep:
        return {
            "thresholds": [m.__dict__ for m in threshold_sweep_from_probs(probs, labels, loss)],
            "default": binary_metrics_from_probs(probs, labels, loss, threshold=threshold).__dict__,
        }
    return binary_metrics_from_probs(probs, labels, loss, threshold=threshold).__dict__


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AITEXPatchDataset(resolve_aitex_dir(args.project_root))
    splits = make_splits(dataset, seed=args.seed)
    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = OpticalStudentClassifier(
        optical_kernels=args.optical_kernels,
        kernel_size=args.kernel_size,
        pooled_size=args.pooled_size,
        hidden_dim=args.hidden_dim,
        optical_activation=args.optical_activation,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    result = {
        "checkpoint": str(args.checkpoint),
        "train": evaluate(model, train_loader, device, args.input_size, args.threshold, args.threshold_sweep),
        "val": evaluate(model, val_loader, device, args.input_size, args.threshold, args.threshold_sweep),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
