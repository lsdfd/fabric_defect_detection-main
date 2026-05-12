from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FabricPSFConfig:
    wavelength_nm: float = 532.0
    grid_pitch_nm: float = 586.0
    detector_distance_mm: float = 2.4
    sim_size: int = 1600
    scale: int = 2
    kernel_index: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare single-channel target PSFs and initial backprop phases for fabric metasurface design."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to fabric_r1_kernels_for_metasurface.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-index", type=int, default=0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--grid-pitch-nm", type=float, default=586.0)
    parser.add_argument("--detector-distance-mm", type=float, default=2.4)
    parser.add_argument("--save-figures", action="store_true")
    return parser.parse_args()


def angular_spectrum_backprop(field: np.ndarray, z_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """Backpropagate a desired detector-plane intensity template to get an initial phase guess."""
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


def normalize_pos_neg(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_pos = float(np.max(positive))
    max_neg = float(np.max(negative))
    denom = max(max_pos, max_neg, 1e-12)
    return positive / denom, negative / denom


def save_preview(output_dir: Path, positive: np.ndarray, negative: np.ndarray, phase_pos: np.ndarray, phase_neg: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(positive, cmap="gray")
    axes[0, 0].set_title("Positive Target PSF")
    axes[0, 1].imshow(negative, cmap="gray")
    axes[0, 1].set_title("Negative Target PSF")
    axes[1, 0].imshow(phase_pos, cmap="twilight")
    axes[1, 0].set_title("Positive Backphase")
    axes[1, 1].imshow(phase_neg, cmap="twilight")
    axes[1, 1].set_title("Negative Backphase")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "preview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = FabricPSFConfig(
        wavelength_nm=args.wavelength_nm,
        grid_pitch_nm=args.grid_pitch_nm,
        detector_distance_mm=args.detector_distance_mm,
        kernel_index=args.kernel_index,
    )

    data = np.load(args.input)
    positive_all = data["positive_expanded"]
    negative_all = data["negative_expanded"]

    if not 0 <= config.kernel_index < len(positive_all):
        raise IndexError(f"kernel_index {config.kernel_index} out of range for {len(positive_all)} kernels")

    positive = np.asarray(positive_all[config.kernel_index], dtype=np.float32)
    negative = np.asarray(negative_all[config.kernel_index], dtype=np.float32)
    positive, negative = normalize_pos_neg(positive, negative)

    wavelength_m = config.wavelength_nm * 1e-9
    pitch_m = config.grid_pitch_nm * 1e-9
    distance_m = config.detector_distance_mm * 1e-3

    backprop_pos = angular_spectrum_backprop(positive, -distance_m, wavelength_m, pitch_m)
    backprop_neg = angular_spectrum_backprop(negative, -distance_m, wavelength_m, pitch_m)
    backphase_pos = np.angle(backprop_pos).astype(np.float32)
    backphase_neg = np.angle(backprop_neg).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "target_psf_and_backphase.npz",
        positive_target=positive,
        negative_target=negative,
        positive_backphase=backphase_pos,
        negative_backphase=backphase_neg,
    )
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    if args.save_figures:
        save_preview(args.output_dir, positive, negative, backphase_pos, backphase_neg)

    print(f"Saved target package to {args.output_dir}")
    print(f"positive_target shape: {positive.shape}")
    print(f"negative_target shape: {negative.shape}")


if __name__ == "__main__":
    main()
