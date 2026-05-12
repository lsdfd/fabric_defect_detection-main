from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FabricMetasurfaceConfig:
    kernel_package: Path
    output_dir: Path
    kernel_index: int = 0
    wavelength_nm: float = 532.0
    grid_pitch_nm: float = 586.0
    detector_distance_mm: float = 2.4
    sim_size: int = 1600
    scale: int = 2
    radius_min_um: float = 0.0
    radius_max_um: float = 0.240
    lookup_table_path: Path | None = None
