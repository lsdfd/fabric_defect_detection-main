from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.data import AITEXPatchDataset, make_balanced_sampler, make_splits, resolve_aitex_dir
from fdd.models import OpticalStudentClassifier, load_teacher
from fdd.training import binary_metrics_from_probs, distillation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a one-convolution optical student with KD.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for supervised task loss.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--optical-kernels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooled-size", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            logits = model.forward_logits(images)
            probs = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    return binary_metrics_from_probs(
        torch.cat(all_probs).view(-1),
        torch.cat(all_labels).view(-1),
        torch.tensor(total_loss / max(total_count, 1)),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = args.project_root
    checkpoint = args.teacher_checkpoint
    if checkpoint is None:
        trained_teacher = project_root / "outputs" / "teacher" / "binary_classifier_best.pt"
        checkpoint = trained_teacher if trained_teacher.exists() else project_root / "models" / "bigger_binary_F1_0.98.pth"
    output_dir = args.output_dir or project_root / "outputs" / "student_kd"
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((224, 224))])
    dataset = AITEXPatchDataset(resolve_aitex_dir(project_root), transform=transform)
    splits = make_splits(dataset, seed=args.seed)
    sampler = None if args.no_balanced_sampler else make_balanced_sampler(splits.train, dataset.labels)
    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)

    teacher = load_teacher(str(checkpoint), device=device)
    student = OpticalStudentClassifier(
        optical_kernels=args.optical_kernels,
        kernel_size=args.kernel_size,
        pooled_size=args.pooled_size,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        running = {"total": 0.0, "task": 0.0, "kd": 0.0}
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            with torch.no_grad():
                teacher_probs = teacher(images)
            student_logits = student.forward_logits(images)
            loss, task_loss, kd_loss = distillation_loss(
                student_logits,
                labels,
                teacher_probs,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running["total"] += loss.item() * batch_size
            running["task"] += task_loss.item() * batch_size
            running["kd"] += kd_loss.item() * batch_size

        train_count = len(splits.train)
        val_metrics = evaluate(student, val_loader, device)
        row = {
            "epoch": epoch,
            "train_total_loss": running["total"] / train_count,
            "train_task_loss": running["task"] / train_count,
            "train_kd_loss": running["kd"] / train_count,
            "val": val_metrics.__dict__,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            torch.save(student.state_dict(), output_dir / "student_best.pt")
            torch.save(student.optical_kernels(), output_dir / "student_optical_kernels.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
