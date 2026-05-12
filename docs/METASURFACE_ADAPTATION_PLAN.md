# Metasurface Adaptation Plan

This document aligns the current fabric student with the existing
`TF_for_PSF_Engineering_CIFAR.ipynb` workflow under
`卷积核->超表面相位设计代码/`.

## Current Starting Point

- best student baseline: `R1`
- student input size: `64 x 64`
- optical frontend: `16` kernels of size `7 x 7`
- channel count: single-channel grayscale

Available artifacts:

- `outputs/remote_pull/student_baseline_R1_seed42/student_best.pt`
- `outputs/remote_pull/student_baseline_R1_seed42/student_optical_kernels.pt`

## Reference Notebook Assumptions

The current TensorFlow notebook is written for the RGB optical encoder paper.
Its assumptions include:

- kernels shape is `16 x 3 x 7 x 7`
- RGB wavelengths are optimized jointly
- each kernel is split into positive and negative parts
- desired PSFs are expanded and padded to a large simulation canvas
- phase is optimized and then mapped to structure parameters using lookup tables

## Fabric-Specific Differences

For the current fabric task we must adapt:

1. RGB kernels -> grayscale kernels
2. `16 x 3 x 7 x 7` -> `16 x 1 x 7 x 7`
3. multi-wavelength target -> first-pass single-wavelength target
4. CIFAR image semantics -> fabric texture binary classification semantics
5. keep positive / negative split, because optical intensity is nonnegative

## Execution Plan

### Version A: Kernel Packaging

Goal:

- export and normalize the current fabric kernels in a clean format
- create positive / negative split
- create expanded and padded target PSFs

Implementation:

- `scripts/export/prepare_fabric_kernels_for_metasurface.py`

Output:

- `outputs/metasurface/fabric_r1_kernels_for_metasurface.npz`

### Version B: Notebook Adaptation

Goal:

- duplicate the CIFAR notebook into a fabric-specific notebook
- replace RGB kernel loading with grayscale kernel loading
- simplify expand / pad logic to one channel
- keep the same phase optimization backbone

Expected new artifact:

- `卷积核->超表面相位设计代码/TF_for_PSF_Engineering_Fabric_R1.ipynb`

### Version C: Fabric Optical Design Refinement

Goal:

- tune enlargement factor, simulation size, and target PSF scaling
- decide whether single-wavelength is sufficient for first-pass proof of concept
- later consider multi-wavelength extension only if justified

## Immediate Next Step

1. package `R1` kernels into metasurface-ready `.npz`
2. inspect resulting target PSFs
3. clone and adapt the CIFAR notebook for grayscale fabric kernels
