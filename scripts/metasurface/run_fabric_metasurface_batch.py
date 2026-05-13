from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from optimize_single_kernel_metasurface import optimize_stage1_target
from port_fabric_metasurface_stage1 import angular_spectrum_backprop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage-1 and positive/negative metasurface optimization for all fabric student kernels."
    )
    parser.add_argument("--kernel-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kernel-indices", type=int, nargs="*", default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--radius-min-um", type=float, default=0.0)
    parser.add_argument("--radius-max-um", type=float, default=0.240)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--grid-pitch-nm", type=float, default=586.0)
    parser.add_argument("--detector-distance-mm", type=float, default=2.4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def prepare_stage1(stage1_dir: Path, kernel_package: Path, kernel_index: int, wavelength_nm: float, grid_pitch_nm: float, detector_distance_mm: float, radius_min_um: float, radius_max_um: float) -> dict:
    data = np.load(kernel_package)
    positive_all = data["positive_expanded"]
    negative_all = data["negative_expanded"]
    raw_kernels = data["kernels"]

    positive = positive_all[kernel_index].astype(np.float32)
    negative = negative_all[kernel_index].astype(np.float32)
    raw_kernel = raw_kernels[kernel_index].astype(np.float32)

    wavelength_m = wavelength_nm * 1e-9
    pitch_m = grid_pitch_nm * 1e-9
    distance_m = detector_distance_mm * 1e-3

    backprop_pos = angular_spectrum_backprop(positive, -distance_m, wavelength_m, pitch_m)
    backprop_neg = angular_spectrum_backprop(negative, -distance_m, wavelength_m, pitch_m)
    backphase_pos = np.angle(backprop_pos).astype(np.float32)
    backphase_neg = np.angle(backprop_neg).astype(np.float32)

    stage1_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage1_dir / "stage1_targets.npz",
        raw_kernel=raw_kernel,
        positive_target=positive,
        negative_target=negative,
        positive_backphase=backphase_pos,
        negative_backphase=backphase_neg,
    )

    summary = {
        "kernel_index": kernel_index,
        "kernel_package": str(kernel_package),
        "wavelength_nm": wavelength_nm,
        "grid_pitch_nm": grid_pitch_nm,
        "detector_distance_mm": detector_distance_mm,
        "radius_min_um": radius_min_um,
        "radius_max_um": radius_max_um,
    }
    (stage1_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def resolve_kernel_indices(kernel_package: Path, requested: list[int] | None) -> list[int]:
    data = np.load(kernel_package)
    total = int(data["kernels"].shape[0])
    if requested is None or len(requested) == 0:
        return list(range(total))
    for idx in requested:
        if idx < 0 or idx >= total:
            raise ValueError(f"Kernel index {idx} out of range [0, {total - 1}]")
    return requested


def main() -> None:
    args = parse_args()
    kernel_indices = resolve_kernel_indices(args.kernel_package, args.kernel_indices)

    batch_summary: list[dict] = []
    for kernel_index in kernel_indices:
        kernel_dir = args.output_root / f"kernel_{kernel_index:02d}"
        stage1_dir = kernel_dir / "stage1"
        prepare_stage1(
            stage1_dir=stage1_dir,
            kernel_package=args.kernel_package,
            kernel_index=kernel_index,
            wavelength_nm=args.wavelength_nm,
            grid_pitch_nm=args.grid_pitch_nm,
            detector_distance_mm=args.detector_distance_mm,
            radius_min_um=args.radius_min_um,
            radius_max_um=args.radius_max_um,
        )

        positive_summary = optimize_stage1_target(
            stage1_dir=stage1_dir,
            target_name="positive",
            iterations=args.iterations,
            lr=args.lr,
            radius_min_um=args.radius_min_um,
            radius_max_um=args.radius_max_um,
            wavelength_nm=args.wavelength_nm,
            grid_pitch_nm=args.grid_pitch_nm,
            detector_distance_mm=args.detector_distance_mm,
            device_arg=args.device,
            roi_size=args.roi_size,
        )
        negative_summary = optimize_stage1_target(
            stage1_dir=stage1_dir,
            target_name="negative",
            iterations=args.iterations,
            lr=args.lr,
            radius_min_um=args.radius_min_um,
            radius_max_um=args.radius_max_um,
            wavelength_nm=args.wavelength_nm,
            grid_pitch_nm=args.grid_pitch_nm,
            detector_distance_mm=args.detector_distance_mm,
            device_arg=args.device,
            roi_size=args.roi_size,
        )

        kernel_summary = {
            "kernel_index": kernel_index,
            "stage1_dir": str(stage1_dir),
            "positive_final_loss": positive_summary["final_loss"],
            "negative_final_loss": negative_summary["final_loss"],
            "roi_bounds": positive_summary["roi_bounds"],
        }
        batch_summary.append(kernel_summary)
        print(json.dumps(kernel_summary, ensure_ascii=False))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "batch_summary.json").write_text(
        json.dumps(
            {
                "kernel_package": str(args.kernel_package),
                "kernel_indices": kernel_indices,
                "iterations": args.iterations,
                "learning_rate": args.lr,
                "roi_size": args.roi_size,
                "results": batch_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved batch summary to {args.output_root / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
