#!/usr/bin/env bash
# Absorbing-mask completion sweep on the MDLM and CDLM checkpoints
# (mask ratio 0.3-0.6, llada_sample steps 1-16). Input of
# sudoku_absorbing_*_comparison.pdf.
# 384 evaluation processes per checkpoint; exits non-zero if any fails.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_sweep completion "${RESULTS_ROOT}/completion" "${MODES:-absorbing}"
