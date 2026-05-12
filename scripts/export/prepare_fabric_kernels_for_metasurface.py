from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare single-channel fabric student kernels for metasurface PSF design."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to student_optical_kernels.pt")
    parser.add_argument("--output", type=Path, required=True, help="Path to output .npz file")
    parser.add_argument("--scale", type=int, default=2, help="Desired PSF enlargement factor")
    parser.add_argument("--sim-size", type=int, default=1600, help="Square simulation canvas size")
    return parser.parse_args()


def expand_kernel(kernel_2d: np.ndarray, scale: int, sim_size: int) -> np.ndarray:
    expanded = np.kron(kernel_2d, np.ones((scale, scale), dtype=np.float32))
    canvas = np.zeros((sim_size, sim_size), dtype=np.float32)
    h, w = expanded.shape
    if h > sim_size or w > sim_size:
        raise ValueError(f"Expanded kernel shape {expanded.shape} exceeds simulation size {sim_size}.")
    start_y = (sim_size - h) // 2
    start_x = (sim_size - w) // 2
    canvas[start_y : start_y + h, start_x : start_x + w] = expanded
    return canvas


def main() -> None:
    args = parse_args()
    kernels = torch.load(args.input, map_location="cpu")
    if isinstance(kernels, torch.Tensor):
        kernels = kernels.detach().cpu().numpy()
    kernels = np.asarray(kernels, dtype=np.float32)

    if kernels.ndim != 4 or kernels.shape[1] != 1:
        raise ValueError(f"Expected kernel shape [N,1,H,W], got {kernels.shape}")

    kernels = np.squeeze(kernels, axis=1)
    positive = np.clip(kernels, 0, None)
    negative = np.clip(-kernels, 0, None)

    positive_expanded = np.stack(
        [expand_kernel(kernel, scale=args.scale, sim_size=args.sim_size) for kernel in positive],
        axis=0,
    )
    negative_expanded = np.stack(
        [expand_kernel(kernel, scale=args.scale, sim_size=args.sim_size) for kernel in negative],
        axis=0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        kernels=kernels,
        positive=positive,
        negative=negative,
        positive_expanded=positive_expanded,
        negative_expanded=negative_expanded,
        scale=np.array(args.scale, dtype=np.int32),
        sim_size=np.array(args.sim_size, dtype=np.int32),
    )
    print(f"Saved metasurface-ready kernel package to {args.output}")
    print(f"kernels shape: {kernels.shape}")
    print(f"positive_expanded shape: {positive_expanded.shape}")


if __name__ == "__main__":
    main()
