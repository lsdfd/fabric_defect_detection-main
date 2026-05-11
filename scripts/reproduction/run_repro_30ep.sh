#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
mkdir -p outputs/notebook_repro_30ep

conda run -n metamat python scripts/reproduce_binary_patch_notebook.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.001 \
  --antialias false \
  --num-workers 0 \
  --save-each-epoch \
  --output-dir outputs/notebook_repro_30ep
