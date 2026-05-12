from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGE1_BASELINES = {
    "R1": {"student_input_size": 64, "optical_kernels": 16, "kernel_size": 7, "pooled_size": 6, "hidden_dim": 256},
    "R2": {"student_input_size": 96, "optical_kernels": 16, "kernel_size": 7, "pooled_size": 6, "hidden_dim": 256},
    "R3": {"student_input_size": 128, "optical_kernels": 16, "kernel_size": 7, "pooled_size": 6, "hidden_dim": 256},
    "R4": {"student_input_size": 96, "optical_kernels": 8, "kernel_size": 7, "pooled_size": 6, "hidden_dim": 256},
    "R5": {"student_input_size": 96, "optical_kernels": 16, "kernel_size": 11, "pooled_size": 6, "hidden_dim": 256},
}

STAGE2_KD = {
    "K1": {"alpha": 0.9, "temperature": 2.0},
    "K2": {"alpha": 0.8, "temperature": 2.0},
    "K3": {"alpha": 0.8, "temperature": 4.0},
    "K4": {"alpha": 0.7, "temperature": 2.0},
    "K5": {"alpha": 0.7, "temperature": 4.0},
}

STAGE3_BINARY_KD_FIX = {
    "B50": {
        "mode": "baseline",
        "epochs": 50,
        "auto_pos_weight": True,
    },
    "L90": {
        "mode": "kd",
        "epochs": 50,
        "alpha": 0.9,
        "temperature": 2.0,
        "kd_target": "logit",
        "auto_pos_weight": True,
    },
    "L95": {
        "mode": "kd",
        "epochs": 50,
        "alpha": 0.95,
        "temperature": 2.0,
        "kd_target": "logit",
        "auto_pos_weight": True,
    },
}

STAGE4_BINARY_KD_ABLATION = {
    "B50N": {
        "mode": "baseline",
        "epochs": 50,
        "auto_pos_weight": False,
    },
    "P90N": {
        "mode": "kd",
        "epochs": 50,
        "alpha": 0.9,
        "temperature": 2.0,
        "kd_target": "prob",
        "auto_pos_weight": False,
    },
    "P95N": {
        "mode": "kd",
        "epochs": 50,
        "alpha": 0.95,
        "temperature": 2.0,
        "kd_target": "prob",
        "auto_pos_weight": False,
    },
    "L95N": {
        "mode": "kd",
        "epochs": 50,
        "alpha": 0.95,
        "temperature": 2.0,
        "kd_target": "logit",
        "auto_pos_weight": False,
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate student baseline / KD sweep commands.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--stage",
        choices=["baseline", "kd", "binary-kd-fix", "binary-kd-ablation"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--teacher-checkpoint", type=str, default="models/bigger_binary_F1_0.98 (1).pth")
    parser.add_argument("--baseline-ids", nargs="*", default=["R1", "R2", "R3", "R4", "R5"])
    parser.add_argument("--kd-ids", nargs="*", default=["K2", "K3", "K4"])
    parser.add_argument("--selected-baselines", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_baseline_command(args: argparse.Namespace, baseline_id: str) -> dict:
    cfg = STAGE1_BASELINES[baseline_id]
    output_dir = (
        args.project_root
        / "outputs"
        / f"student_baseline_{baseline_id}_seed{args.seed}"
    )
    cmd = [
        "python",
        "scripts/training/train_student_kd.py",
        "--mode",
        "baseline",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optical-kernels",
        str(cfg["optical_kernels"]),
        "--kernel-size",
        str(cfg["kernel_size"]),
        "--pooled-size",
        str(cfg["pooled_size"]),
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--student-input-size",
        str(cfg["student_input_size"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    return {"id": baseline_id, "type": "baseline", "config": cfg, "output_dir": str(output_dir), "command": cmd}


def build_kd_command(args: argparse.Namespace, baseline_id: str, kd_id: str) -> dict:
    base_cfg = STAGE1_BASELINES[baseline_id]
    kd_cfg = STAGE2_KD[kd_id]
    output_dir = (
        args.project_root
        / "outputs"
        / f"student_kd_{baseline_id}_{kd_id}_seed{args.seed}"
    )
    cmd = [
        "python",
        "scripts/training/train_student_kd.py",
        "--mode",
        "kd",
        "--teacher-checkpoint",
        args.teacher_checkpoint,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optical-kernels",
        str(base_cfg["optical_kernels"]),
        "--kernel-size",
        str(base_cfg["kernel_size"]),
        "--pooled-size",
        str(base_cfg["pooled_size"]),
        "--hidden-dim",
        str(base_cfg["hidden_dim"]),
        "--student-input-size",
        str(base_cfg["student_input_size"]),
        "--alpha",
        str(kd_cfg["alpha"]),
        "--temperature",
        str(kd_cfg["temperature"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    return {
        "id": f"{baseline_id}_{kd_id}",
        "type": "kd",
        "baseline": baseline_id,
        "kd": kd_id,
        "config": {**base_cfg, **kd_cfg},
        "output_dir": str(output_dir),
        "command": cmd,
    }


def build_binary_kd_fix_command(args: argparse.Namespace, baseline_id: str, variant_id: str) -> dict:
    base_cfg = STAGE1_BASELINES[baseline_id]
    variant_cfg = STAGE3_BINARY_KD_FIX[variant_id]
    mode = variant_cfg["mode"]
    output_dir = (
        args.project_root
        / "outputs"
        / f"student_{mode}_{baseline_id}_{variant_id}_seed{args.seed}"
    )
    cmd = [
        "python",
        "scripts/training/train_student_kd.py",
        "--mode",
        mode,
        "--epochs",
        str(variant_cfg["epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optical-kernels",
        str(base_cfg["optical_kernels"]),
        "--kernel-size",
        str(base_cfg["kernel_size"]),
        "--pooled-size",
        str(base_cfg["pooled_size"]),
        "--hidden-dim",
        str(base_cfg["hidden_dim"]),
        "--student-input-size",
        str(base_cfg["student_input_size"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    if mode == "kd":
        cmd.extend(
            [
                "--teacher-checkpoint",
                args.teacher_checkpoint,
                "--alpha",
                str(variant_cfg["alpha"]),
                "--temperature",
                str(variant_cfg["temperature"]),
                "--kd-target",
                str(variant_cfg["kd_target"]),
            ]
        )
    if variant_cfg.get("auto_pos_weight"):
        cmd.append("--auto-pos-weight")

    return {
        "id": f"{baseline_id}_{variant_id}",
        "type": "binary-kd-fix",
        "baseline": baseline_id,
        "variant": variant_id,
        "config": {**base_cfg, **variant_cfg},
        "output_dir": str(output_dir),
        "command": cmd,
    }


def build_binary_kd_ablation_command(args: argparse.Namespace, baseline_id: str, variant_id: str) -> dict:
    base_cfg = STAGE1_BASELINES[baseline_id]
    variant_cfg = STAGE4_BINARY_KD_ABLATION[variant_id]
    mode = variant_cfg["mode"]
    output_dir = (
        args.project_root
        / "outputs"
        / f"student_{mode}_{baseline_id}_{variant_id}_seed{args.seed}"
    )
    cmd = [
        "python",
        "scripts/training/train_student_kd.py",
        "--mode",
        mode,
        "--epochs",
        str(variant_cfg["epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optical-kernels",
        str(base_cfg["optical_kernels"]),
        "--kernel-size",
        str(base_cfg["kernel_size"]),
        "--pooled-size",
        str(base_cfg["pooled_size"]),
        "--hidden-dim",
        str(base_cfg["hidden_dim"]),
        "--student-input-size",
        str(base_cfg["student_input_size"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    if mode == "kd":
        cmd.extend(
            [
                "--teacher-checkpoint",
                args.teacher_checkpoint,
                "--alpha",
                str(variant_cfg["alpha"]),
                "--temperature",
                str(variant_cfg["temperature"]),
                "--kd-target",
                str(variant_cfg["kd_target"]),
            ]
        )
    if variant_cfg.get("auto_pos_weight"):
        cmd.append("--auto-pos-weight")

    return {
        "id": f"{baseline_id}_{variant_id}",
        "type": "binary-kd-ablation",
        "baseline": baseline_id,
        "variant": variant_id,
        "config": {**base_cfg, **variant_cfg},
        "output_dir": str(output_dir),
        "command": cmd,
    }


def main() -> None:
    args = parse_args()

    if args.stage == "baseline":
        jobs = [build_baseline_command(args, baseline_id) for baseline_id in args.baseline_ids]
    elif args.stage == "kd":
        selected = args.selected_baselines or ["R2", "R5"]
        jobs = [
            build_kd_command(args, baseline_id, kd_id)
            for baseline_id in selected
            for kd_id in args.kd_ids
        ]
    elif args.stage == "binary-kd-fix":
        selected = args.selected_baselines or ["R2"]
        jobs = [
            build_binary_kd_fix_command(args, baseline_id, variant_id)
            for baseline_id in selected
            for variant_id in ["B50", "L90", "L95"]
        ]
    else:
        selected = args.selected_baselines or ["R2"]
        jobs = [
            build_binary_kd_ablation_command(args, baseline_id, variant_id)
            for baseline_id in selected
            for variant_id in ["B50N", "P90N", "P95N", "L95N"]
        ]

    payload = {
        "stage": args.stage,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "jobs": jobs,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Wrote sweep manifest to {args.output}")
        return

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
