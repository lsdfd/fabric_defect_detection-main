# Metasurface Porting Notes

This note records how the CIFAR RGB notebook should be ported into the current
fabric defect workflow.

## What Can Be Reused Directly

- angular spectrum propagation idea
- positive / negative kernel split
- enlarged PSF target construction
- phase-to-structure lookup based optimization logic
- radius / width constrained optimization loop

## What Must Change

### 1. Kernel Tensor Convention

Reference notebook:

- `16 x 3 x 7 x 7`

Current fabric student:

- `16 x 1 x 7 x 7`

So the first porting step is not physics, but data convention cleanup.

### 2. Color Channels

Reference notebook jointly optimizes:

- red: `635 nm`
- green: `532 nm`
- blue: `450 nm`

Current first-pass fabric setting should use:

- single wavelength
- recommended first-pass: `532 nm`

This keeps the optimization path simpler and more faithful to the grayscale
student.

### 3. Input / Task Semantics

Reference scenario:

- CIFAR image classification
- RGB visual semantics

Current scenario:

- grayscale fabric texture discrimination
- binary defect / non-defect student

This affects how much enlargement, blur tolerance, and PSF interpretability we
need to preserve.

### 4. Practical Migration Strategy

#### Stage A

- prepare `R1` kernels as a clean `.npz`
- confirm positive / negative split
- confirm expanded detector-plane target PSFs

#### Stage B

- scriptify detector-plane target preparation and backphase initialization
- this is now implemented in:
  - `scripts/metasurface/prepare_fabric_psf_targets.py`

#### Stage C

- duplicate the CIFAR notebook into a fabric-specific notebook
- replace RGB loading and visualization cells
- route all target generation through the new fabric `.npz`

#### Stage D

- later decide whether to:
  - stay single-wavelength
  - or recover a multi-wavelength optical design path

## Current Recommendation

Do not port the whole RGB notebook blindly.

Instead:

1. reuse the physical optimization backbone
2. replace the target preparation path first
3. validate one kernel end-to-end
4. only then scale to all 16 positive / negative kernels
