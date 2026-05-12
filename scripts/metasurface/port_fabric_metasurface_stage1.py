from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fabric_metasurface_config import FabricMetasurfaceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 fabric metasurface port: single-channel, single-wavelength, single-kernel target preparation."
    )
    parser.add_argument("--kernel-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-index", type=int, default=0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--grid-pitch-nm", type=float, default=586.0)
    parser.add_argument("--detector-distance-mm", type=float, default=2.4)
    parser.add_argument("--sim-size", type=int, default=1600)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--radius-min-um", type=float, default=0.0)
    parser.add_argument("--radius-max-um", type=float, default=0.240)
    parser.add_argument("--lookup-table-path", type=Path, default=None)
    return parser.parse_args()


def angular_spectrum_backprop(field: np.ndarray, z_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    e = np.asarray(field, dtype=np.complex64)
    if np.sign(z_m) < 0:
        e = np.conjugate(e)

    spectrum = np.fft.fft2(e)
    nx, ny = e.shape
    u = np.fft.fftfreq(nx, d=pitch_m)
    v = np.fft.fftfreq(ny, d=pitch_m)
    v_grid, u_grid = np.meshgrid(v, u)
    w = np.sqrt(0j + 1.0 / wavelength_m**2 - u_grid**2 - v_grid**2).real
    propagated = spectrum * np.exp(1j * 2 * np.pi * w * abs(z_m))
    return np.fft.ifft2(propagated)


def save_preview(output_dir: Path, positive: np.ndarray, negative: np.ndarray, phase_pos: np.ndarray, phase_neg: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(positive, cmap="gray")
    axes[0, 0].set_title("Positive Target")
    axes[0, 1].imshow(negative, cmap="gray")
    axes[0, 1].set_title("Negative Target")
    axes[1, 0].imshow(phase_pos, cmap="twilight")
    axes[1, 0].set_title("Positive Backphase")
    axes[1, 1].imshow(phase_neg, cmap="twilight")
    axes[1, 1].set_title("Negative Backphase")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_preview.png", dpi=180)
    plt.close(fig)


def summarize_lookup_table(lookup_table_path: Path | None) -> dict:
    if lookup_table_path is None:
        return {
            "status": "missing",
            "message": "No RCWA/lookup table path provided yet. Stage-1 stops before phase-to-radius optimization.",
        }
    if not lookup_table_path.exists():
        return {
            "status": "not_found",
            "message": f"Lookup table path does not exist: {lookup_table_path}",
        }
    data = np.loadtxt(lookup_table_path)
    return {
        "status": "available",
        "path": str(lookup_table_path),
        "shape": list(data.shape),
        "radius_min_um": float(np.min(data[:, 0])),
        "radius_max_um": float(np.max(data[:, 0])),
    }


def stringify_paths(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: stringify_paths(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [stringify_paths(value) for value in obj]
    return obj


def main() -> None:
    args = parse_args()
    config = FabricMetasurfaceConfig(
        kernel_package=args.kernel_package,
        output_dir=args.output_dir,
        kernel_index=args.kernel_index,
        wavelength_nm=args.wavelength_nm,
        grid_pitch_nm=args.grid_pitch_nm,
        detector_distance_mm=args.detector_distance_mm,
        sim_size=args.sim_size,
        scale=args.scale,
        radius_min_um=args.radius_min_um,
        radius_max_um=args.radius_max_um,
        lookup_table_path=args.lookup_table_path,
    )

    data = np.load(config.kernel_package)
    positive_all = data["positive_expanded"]
    negative_all = data["negative_expanded"]
    raw_kernels = data["kernels"]

    positive = positive_all[config.kernel_index].astype(np.float32)
    negative = negative_all[config.kernel_index].astype(np.float32)
    raw_kernel = raw_kernels[config.kernel_index].astype(np.float32)

    wavelength_m = config.wavelength_nm * 1e-9
    pitch_m = config.grid_pitch_nm * 1e-9
    distance_m = config.detector_distance_mm * 1e-3

    backprop_pos = angular_spectrum_backprop(positive, -distance_m, wavelength_m, pitch_m)
    backprop_neg = angular_spectrum_backprop(negative, -distance_m, wavelength_m, pitch_m)
    backphase_pos = np.angle(backprop_pos).astype(np.float32)
    backphase_neg = np.angle(backprop_neg).astype(np.float32)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "stage1_targets.npz",
        raw_kernel=raw_kernel,
        positive_target=positive,
        negative_target=negative,
        positive_backphase=backphase_pos,
        negative_backphase=backphase_neg,
    )
    save_preview(config.output_dir, positive, negative, backphase_pos, backphase_neg)

    summary = {
        "config": stringify_paths(asdict(config)),
        "lookup_table": summarize_lookup_table(config.lookup_table_path),
        "notes": [
            "This stage adapts the RGB notebook to grayscale fabric targets.",
            "Optimization over radius/phase is intentionally deferred until RCWA/lookup data is wired in.",
            "The next step is replacing the notebook's RGB target and wavelength assumptions with this stage-1 package.",
        ],
    }
    (config.output_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved stage-1 fabric metasurface package to {config.output_dir}")


if __name__ == "__main__":
    main()
