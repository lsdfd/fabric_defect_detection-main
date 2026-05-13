from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


GREEN_PHASE_COEFFS = {
    "a": -2.28293,
    "b": 0.01878,
    "c": -0.07134,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a single-channel single-wavelength metasurface for one fabric kernel target."
    )
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["positive", "negative"], default="negative")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--radius-min-um", type=float, default=0.0)
    parser.add_argument("--radius-max-um", type=float, default=0.240)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--grid-pitch-nm", type=float, default=586.0)
    parser.add_argument("--detector-distance-mm", type=float, default=2.4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--roi-size", type=int, default=96)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def phase_cycles_from_radius(radius_um: torch.Tensor) -> torch.Tensor:
    a = GREEN_PHASE_COEFFS["a"]
    b = GREEN_PHASE_COEFFS["b"]
    c = GREEN_PHASE_COEFFS["c"]
    return a * (torch.exp((radius_um * radius_um - 2.0 * b * radius_um) / c) - 1.0)


def phase_radians_from_radius(radius_um: torch.Tensor) -> torch.Tensor:
    return phase_cycles_from_radius(radius_um) * (2.0 * math.pi)


def initialize_radius_from_backphase(backphase_rad: np.ndarray, radius_min_um: float, radius_max_um: float) -> np.ndarray:
    radius_samples = np.linspace(radius_min_um, radius_max_um, 4000, dtype=np.float32)
    phase_cycles = GREEN_PHASE_COEFFS["a"] * (
        np.exp((radius_samples * radius_samples - 2.0 * GREEN_PHASE_COEFFS["b"] * radius_samples) / GREEN_PHASE_COEFFS["c"]) - 1.0
    )
    phase_rad = phase_cycles * (2.0 * np.pi)
    phase_wrapped = np.mod(phase_rad, 2.0 * np.pi)
    target_wrapped = np.mod(backphase_rad, 2.0 * np.pi)

    flat = target_wrapped.reshape(-1, 1)
    distances = np.abs(np.angle(np.exp(1j * (flat - phase_wrapped.reshape(1, -1)))))
    nearest = distances.argmin(axis=1)
    return radius_samples[nearest].reshape(backphase_rad.shape)


def get_frequencies(nx: int, ny: int, dx: float, dy: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * (2.0 * math.pi)
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * (2.0 * math.pi)
    k_y, k_x = torch.meshgrid(ky, kx, indexing="xy")
    return k_x, k_y


def propagate_angular_spectrum(field: torch.Tensor, wavelength_m: float, z_m: float, dx: float, dy: float) -> torch.Tensor:
    nx, ny = field.shape
    device = field.device
    k = 2.0 * math.pi / wavelength_m
    k_x, k_y = get_frequencies(nx, ny, dx, dy, device)
    kz_squared = (k**2 - k_x**2 - k_y**2).to(torch.complex64)
    k_z = torch.sqrt(kz_squared)
    spectrum = torch.fft.fft2(field)
    propagator = torch.exp(1j * k_z * z_m)
    return torch.fft.ifft2(spectrum * propagator)


def normalize_l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x).clamp_min(1e-12)


def centered_roi_bounds(array: np.ndarray, roi_size: int) -> tuple[int, int, int, int]:
    nonzero = np.argwhere(array > 1e-8)
    height, width = array.shape
    if nonzero.size == 0:
        center_y = height // 2
        center_x = width // 2
    else:
        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

    roi_size = min(roi_size, height, width)
    half = roi_size // 2
    top = max(0, center_y - half)
    left = max(0, center_x - half)
    bottom = min(height, top + roi_size)
    right = min(width, left + roi_size)
    top = max(0, bottom - roi_size)
    left = max(0, right - roi_size)
    return top, bottom, left, right


def crop_array(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bounds
    return array[top:bottom, left:right]


def embed_array(cropped: np.ndarray, full_shape: tuple[int, int], bounds: tuple[int, int, int, int]) -> np.ndarray:
    full = np.zeros(full_shape, dtype=cropped.dtype)
    top, bottom, left, right = bounds
    full[top:bottom, left:right] = cropped
    return full


def save_preview(output_dir: Path, target: np.ndarray, simulated: np.ndarray, phase: np.ndarray, radius: np.ndarray, losses: list[float]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(target, cmap="gray")
    axes[0, 0].set_title("Target PSF")
    axes[0, 1].imshow(simulated, cmap="gray")
    axes[0, 1].set_title("Simulated PSF")
    axes[0, 2].imshow(simulated - target, cmap="bwr")
    axes[0, 2].set_title("Difference")
    axes[1, 0].imshow(phase, cmap="twilight")
    axes[1, 0].set_title("Optimized Phase")
    axes[1, 1].imshow(radius, cmap="viridis")
    axes[1, 1].set_title("Optimized Radius")
    axes[1, 2].plot(losses)
    axes[1, 2].set_title("Loss Curve")
    for ax in axes.ravel():
        ax.axis("off" if ax is not axes[1, 2] else "on")
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_preview.png", dpi=180)
    plt.close(fig)


def optimize_stage1_target(
    stage1_dir: Path,
    target_name: str,
    iterations: int,
    lr: float,
    radius_min_um: float,
    radius_max_um: float,
    wavelength_nm: float,
    grid_pitch_nm: float,
    detector_distance_mm: float,
    device_arg: str,
    roi_size: int,
) -> dict:
    device = choose_device(device_arg)
    target_package = np.load(stage1_dir / "stage1_targets.npz")
    summary_path = stage1_dir / "stage1_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if target_name == "positive":
        target_np = target_package["positive_target"].astype(np.float32)
        backphase_np = target_package["positive_backphase"].astype(np.float32)
    else:
        target_np = target_package["negative_target"].astype(np.float32)
        backphase_np = target_package["negative_backphase"].astype(np.float32)

    roi_bounds = centered_roi_bounds(target_np, roi_size)
    target_roi_np = crop_array(target_np, roi_bounds)
    backphase_roi_np = crop_array(backphase_np, roi_bounds)

    init_radius_np = initialize_radius_from_backphase(
        backphase_roi_np,
        radius_min_um=radius_min_um,
        radius_max_um=radius_max_um,
    )

    radius = torch.tensor(init_radius_np, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.tensor(target_roi_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([radius], lr=lr)

    wavelength_m = wavelength_nm * 1e-9
    grid_pitch_m = grid_pitch_nm * 1e-9
    detector_distance_m = detector_distance_mm * 1e-3

    losses: list[float] = []
    for _ in range(iterations):
        optimizer.zero_grad()
        radius_clamped = torch.clamp(radius, radius_min_um, radius_max_um)
        phase = phase_radians_from_radius(radius_clamped)
        field = torch.exp(1j * phase.to(torch.complex64))
        propagated = propagate_angular_spectrum(field, wavelength_m, detector_distance_m, grid_pitch_m, grid_pitch_m)
        intensity = torch.abs(propagated) ** 2
        simulated = normalize_l2(intensity)
        desired = normalize_l2(target)
        loss = torch.mean((simulated - desired) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    radius_final = torch.clamp(radius.detach(), radius_min_um, radius_max_um)
    phase_final = phase_radians_from_radius(radius_final).detach()
    field_final = torch.exp(1j * phase_final.to(torch.complex64))
    propagated_final = propagate_angular_spectrum(field_final, wavelength_m, detector_distance_m, grid_pitch_m, grid_pitch_m)
    intensity_final = normalize_l2(torch.abs(propagated_final) ** 2).detach().cpu().numpy().astype(np.float32)
    full_simulated = embed_array(intensity_final, target_np.shape, roi_bounds)
    full_phase = embed_array(phase_final.cpu().numpy().astype(np.float32), target_np.shape, roi_bounds)
    full_radius = embed_array(radius_final.cpu().numpy().astype(np.float32), target_np.shape, roi_bounds)
    full_init_radius = embed_array(init_radius_np.astype(np.float32), target_np.shape, roi_bounds)

    output_dir = stage1_dir / f"optimized_{target_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "optimized_result.npz",
        target=target_np,
        simulated=full_simulated,
        optimized_radius=full_radius,
        optimized_phase=full_phase,
        loss_history=np.asarray(losses, dtype=np.float32),
        init_radius=full_init_radius,
        roi_target=target_roi_np,
        roi_simulated=intensity_final,
        roi_bounds=np.asarray(roi_bounds, dtype=np.int32),
    )
    save_preview(
        output_dir,
        target_np,
        full_simulated,
        full_phase,
        full_radius,
        losses,
    )

    result_summary = {
        "target": target_name,
        "iterations": iterations,
        "learning_rate": lr,
        "device": str(device),
        "final_loss": losses[-1],
        "roi_size": roi_size,
        "roi_bounds": [int(v) for v in roi_bounds],
        "wavelength_nm": wavelength_nm,
        "grid_pitch_nm": grid_pitch_nm,
        "detector_distance_mm": detector_distance_mm,
        "stage1_summary": summary,
    }
    (output_dir / "result_summary.json").write_text(json.dumps(result_summary, indent=2, ensure_ascii=False))
    return result_summary


def main() -> None:
    args = parse_args()
    result_summary = optimize_stage1_target(
        stage1_dir=args.stage1_dir,
        target_name=args.target,
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
    print(json.dumps(result_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
