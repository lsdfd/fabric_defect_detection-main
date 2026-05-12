from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXPatchDataset, make_balanced_sampler, make_splits, resolve_aitex_dir
from fdd.models import OpticalStudentClassifier, load_teacher
from fdd.training import (
    baseline_student_loss,
    binary_metrics_from_probs,
    distillation_loss,
    threshold_sweep_from_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a one-convolution student baseline or KD model.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["baseline", "kd"], default="kd")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for supervised task loss.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-target", choices=["prob", "logit"], default="prob")
    parser.add_argument("--optical-kernels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooled-size", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--optical-activation", choices=["relu", "identity"], default="relu")
    parser.add_argument("--student-input-size", type=int, default=96)
    parser.add_argument("--teacher-input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-pos-weight", type=float, default=None)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def evaluate(model, loader, device, student_input_size, pos_weight=None):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if student_input_size != 256:
                images = F.interpolate(
                    images,
                    size=(student_input_size, student_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            labels = labels.to(device).view(-1, 1)
            logits = model.forward_logits(images)
            probs = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    probs = torch.cat(all_probs).view(-1)
    labels = torch.cat(all_labels).view(-1)
    loss = torch.tensor(total_loss / max(total_count, 1))
    default_metrics = binary_metrics_from_probs(probs, labels, loss)
    sweep_metrics = threshold_sweep_from_probs(probs, labels, loss)
    best_metrics = max(sweep_metrics, key=lambda metric: metric.f1)
    return {
        "default": default_metrics,
        "best": best_metrics,
        "sweep": sweep_metrics,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = args.project_root
    checkpoint = args.teacher_checkpoint
    if args.mode == "kd":
        if checkpoint is None:
            trained_teacher = project_root / "outputs" / "teacher" / "binary_classifier_best.pt"
            checkpoint = trained_teacher if trained_teacher.exists() else project_root / "models" / "bigger_binary_F1_0.98.pth"
    output_dir = args.output_dir or project_root / "outputs" / (
        f"student_{args.mode}_k{args.optical_kernels}_s{args.kernel_size}_p{args.pooled_size}_h{args.hidden_dim}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AITEXPatchDataset(resolve_aitex_dir(project_root), transform=None)
    splits = make_splits(dataset, seed=args.seed)
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(splits.train, dataset.labels)
    train_labels = [dataset.labels[idx] for idx in splits.train.indices]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = None
    if args.task_pos_weight is not None:
        pos_weight = torch.tensor([args.task_pos_weight], device=device, dtype=torch.float32)
    elif args.auto_pos_weight and pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], device=device, dtype=torch.float32)

    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    teacher = load_teacher(str(checkpoint), device=device) if args.mode == "kd" else None
    student = OpticalStudentClassifier(
        optical_kernels=args.optical_kernels,
        kernel_size=args.kernel_size,
        pooled_size=args.pooled_size,
        hidden_dim=args.hidden_dim,
        optical_activation=args.optical_activation,
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    history = []
    best_f1 = -1.0
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        running = {"total": 0.0, "task": 0.0, "kd": 0.0}
        epoch_started = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            student_images = images
            if args.student_input_size != 256:
                student_images = F.interpolate(
                    student_images,
                    size=(args.student_input_size, args.student_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            labels = labels.to(device).view(-1, 1)
            student_logits = student.forward_logits(student_images)
            if args.mode == "kd":
                with torch.no_grad():
                    teacher_images = F.interpolate(
                        images,
                        size=(args.teacher_input_size, args.teacher_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    teacher_probs = teacher(teacher_images)
                breakdown = distillation_loss(
                    student_logits,
                    labels,
                    teacher_probs,
                    alpha=args.alpha,
                    temperature=args.temperature,
                    kd_target=args.kd_target,
                    pos_weight=pos_weight,
                )
            else:
                breakdown = baseline_student_loss(student_logits, labels, pos_weight=pos_weight)

            optimizer.zero_grad()
            breakdown.total.backward()
            optimizer.step()

            batch_size = images.size(0)
            running["total"] += breakdown.total.item() * batch_size
            running["task"] += breakdown.task.item() * batch_size
            running["kd"] += breakdown.kd.item() * batch_size

        train_count = len(splits.train)
        val_metrics = evaluate(
            student,
            val_loader,
            device,
            student_input_size=args.student_input_size,
            pos_weight=pos_weight,
        )
        row = {
            "epoch": epoch,
            "mode": args.mode,
            "train_total_loss": running["total"] / train_count,
            "train_task_loss": running["task"] / train_count,
            "train_kd_loss": running["kd"] / train_count,
            "epoch_seconds": time.time() - epoch_started,
            "val_default": val_metrics["default"].__dict__,
            "val_best": val_metrics["best"].__dict__,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if val_metrics["best"].f1 > best_f1:
            best_f1 = val_metrics["best"].f1
            torch.save(student.state_dict(), output_dir / "student_best.pt")
            torch.save(student.optical_kernels(), output_dir / "student_optical_kernels.pt")

    torch.save(student.state_dict(), output_dir / "student_last.pt")
    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "teacher_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "history": history,
        "best_val_f1": best_f1,
        "total_seconds": time.time() - started_at,
        "best_checkpoint": str(output_dir / "student_best.pt"),
        "last_checkpoint": str(output_dir / "student_last.pt"),
    }
    (output_dir / "history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
