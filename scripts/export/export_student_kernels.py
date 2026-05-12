from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdd.models import OpticalStudentClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student optical kernels for PSF design.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optical-kernels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooled-size", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--optical-activation", choices=["relu", "identity"], default="relu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = OpticalStudentClassifier(
        optical_kernels=args.optical_kernels,
        kernel_size=args.kernel_size,
        pooled_size=args.pooled_size,
        hidden_dim=args.hidden_dim,
        optical_activation=args.optical_activation,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    kernels = model.optical_kernels().numpy()
    kernels = np.squeeze(kernels, axis=1)

    positive = np.clip(kernels, 0, None)
    negative = np.clip(-kernels, 0, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, kernels=kernels, positive=positive, negative=negative)
    print(f"saved kernels to {args.output} with shape {kernels.shape}")


if __name__ == "__main__":
    main()
