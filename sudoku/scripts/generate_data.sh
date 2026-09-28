#!/usr/bin/env bash
# Generate the Sudoku train/test split (36,000 / 2,000 solved boards, as in the paper).
# Set GEN_WORKERS to limit the number of generator processes (default: all cores).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd "${SUDOKU_DIR}"
python generate_data.py \
    --train-size "${TRAIN_SIZE:-36000}" \
    --test-size "${TEST_SIZE:-2000}" \
    --output-dir "${DATA_DIR}" \
    ${GEN_WORKERS:+--num-workers "${GEN_WORKERS}"}
