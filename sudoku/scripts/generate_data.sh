#!/usr/bin/env bash
# Generate the Sudoku train/test split (36,000 / 2,000 solved boards, as in the paper).
# TRAIN_SIZE and TEST_SIZE change the split sizes; GEN_WORKERS limits the number of
# generator processes (default: all cores). Extra arguments are forwarded to
# generate_data.py, e.g. --force to regenerate a split that already exists.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd "${SUDOKU_DIR}"
python generate_data.py \
    --train-size "${TRAIN_SIZE:-36000}" \
    --test-size "${TEST_SIZE:-2000}" \
    --output-dir "${DATA_DIR}" \
    ${GEN_WORKERS:+--num-workers "${GEN_WORKERS}"} \
    "$@"
