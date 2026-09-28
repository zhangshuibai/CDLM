#!/usr/bin/env bash
# Uniform-noise sweep on the MDLM and CDLM checkpoints:
#   uniform_noise_only       one forward pass, confidence on clean vs corrupted cells
#                            (noise ratio 0.1 / 0.2 / 0.3)
#   uniform_noise_diffusion  editable-region remask refinement
#                            (noise 0.1-0.3 x editable 0.4-0.6, llada_sample steps 1-4)
# Inputs of sudoku_uniform_noise_comparison.pdf and
# sudoku_uniform_noise_diffusion_*_comparison.pdf. The original run also included
# MODES="absorbing" (steps 1-4), which no figure uses.
# 3 + 288 = 291 evaluation processes per checkpoint; exits non-zero if any fails.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_sweep refinement "${RESULTS_ROOT}/refinement" \
    "${MODES:-uniform_noise_only uniform_noise_diffusion}"
