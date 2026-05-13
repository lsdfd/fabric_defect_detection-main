from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXSegmentationPatchDataset, make_splits, resolve_aitex_dir
from fdd.models import OpticalSegmentationStudent, load_unet_teacher
from fdd.unet import (
    MixedLoss,
    notebook_deploy_iou,
    segmentation_baseline_loss,
    segmentation_distillation_loss,
    segmentation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight optical-style segmentation student with KD/NTKD.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["baseline", "kd"], default="kd")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optical-kernels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--backend-channels", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for supervised segmentation loss.")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for dense KD loss.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kd-mode", choices=["logit", "prob"], default="logit")
    parser.add_argument("--ntkd-weight", type=float, default=0.0)
    parser.add_argument("--ntkd-pooled-size", type=int, default=8)
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
            outputs = model.forward_logits(inputs)
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
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = args.project_root
    teacher_checkpoint = args.teacher_checkpoint
    if args.mode == "kd" and teacher_checkpoint is None:
        trained_teacher = project_root / "outputs" / "unet_repro_100ep" / "unet_repro_last.pt"
        teacher_checkpoint = trained_teacher if trained_teacher.exists() else project_root / "models" / "unet_seg_200epoch.pt"

    output_dir = args.output_dir or project_root / "outputs" / (
        f"unet_student_{args.mode}_k{args.optical_kernels}_s{args.kernel_size}_b{args.backend_channels}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AITEXSegmentationPatchDataset(resolve_aitex_dir(project_root), transform=None)
    splits = make_splits(dataset, seed=args.seed)
    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    student = OpticalSegmentationStudent(
        optical_kernels=args.optical_kernels,
        kernel_size=args.kernel_size,
        backend_channels=args.backend_channels,
    ).to(device)
    teacher = load_unet_teacher(str(teacher_checkpoint), device=device) if args.mode == "kd" else None
    criterion = MixedLoss(10, 2)
    optimizer = Adam(student.parameters(), lr=args.lr)

    history = []
    best_val_iou = -1.0
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        running = {"total": 0.0, "task": 0.0, "kd": 0.0, "ntkd": 0.0}
        epoch_started = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            student_logits = student.forward_logits(inputs)
            if args.mode == "kd":
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                breakdown = segmentation_distillation_loss(
                    student_logits=student_logits,
                    targets=targets,
                    teacher_logits=teacher_logits,
                    criterion=criterion,
                    alpha=args.alpha,
                    beta=args.beta,
                    temperature=args.temperature,
                    kd_mode=args.kd_mode,
                    ntkd_weight=args.ntkd_weight,
                    ntkd_pooled_size=args.ntkd_pooled_size,
                )
            else:
                breakdown = segmentation_baseline_loss(student_logits, targets, criterion)

            optimizer.zero_grad()
            breakdown.total.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running["total"] += breakdown.total.item() * batch_size
            running["task"] += breakdown.task.item() * batch_size
            running["kd"] += breakdown.kd.item() * batch_size
            running["ntkd"] += breakdown.ntkd.item() * batch_size

        row = {
            "epoch": epoch,
            "mode": args.mode,
            "train_total_loss": running["total"] / len(splits.train),
            "train_task_loss": running["task"] / len(splits.train),
            "train_kd_loss": running["kd"] / len(splits.train),
            "train_ntkd_loss": running["ntkd"] / len(splits.train),
            "epoch_seconds": time.time() - epoch_started,
        }

        if args.eval_each_epoch or epoch == args.epochs:
            row["train_eval"] = evaluate(student, train_loader, criterion, device)
            row["val_eval"] = evaluate(student, val_loader, criterion, device)

        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if "val_eval" in row and row["val_eval"]["deploy_iou"] > best_val_iou:
            best_val_iou = row["val_eval"]["deploy_iou"]
            torch.save(student.state_dict(), output_dir / "student_best.pt")
            torch.save(student.optical_kernels(), output_dir / "student_optical_kernels.pt")

    torch.save(student.state_dict(), output_dir / "student_last.pt")
    if "val_eval" not in history[-1]:
        history[-1]["train_eval"] = evaluate(student, train_loader, criterion, device)
        history[-1]["val_eval"] = evaluate(student, val_loader, criterion, device)

    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint is not None else None,
        "history": history,
        "best_val_deploy_iou": best_val_iou,
        "total_seconds": time.time() - started_at,
        "best_checkpoint": str(output_dir / "student_best.pt"),
        "last_checkpoint": str(output_dir / "student_last.pt"),
    }
    (output_dir / "history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
