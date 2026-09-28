#!/usr/bin/env bash
# Train the Sudoku DiT from random initialisation (paper Appendix F).
#   bash scripts/train.sh absorbing   # MDLM: absorbing (mask-only) noise
#   bash scripts/train.sh mixture     # CDLM: absorbing + uniform replacement, MIXTURE_PROB (default 0.1)
# Extra arguments are forwarded to train.py. Hyperparameters are those of the
# original run_experiment.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ $# -lt 1 ]; then
    echo "usage: bash scripts/train.sh absorbing|mixture [extra train.py args]" >&2
    exit 1
fi
NOISE="$1"
shift
case "${NOISE}" in
    absorbing)
        LOSS_MODE=absorbing
        OUTPUT_DIR="${ABSORBING_CKPT_DIR}"
        ;;
    mixture)
        LOSS_MODE=uniform_absorbing_mixture
        # train.py appends prob_<MIXTURE_PROB> to this directory.
        OUTPUT_DIR="$(dirname "${MIXTURE_CKPT_DIR}")"
        ;;
    *)
        echo "Unknown noise setting '${NOISE}' (expected absorbing or mixture)" >&2
        exit 1
        ;;
esac

cd "${SUDOKU_DIR}"
python train.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-batch-size 64 \
    --eval-batch-size 2000 \
    --num-steps 200000 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.9 \
    --num-workers "${NUM_WORKERS}" \
    --log-interval 100 \
    --eval-interval 1000 \
    --save-interval 10000 \
    --loss-mode "${LOSS_MODE}" \
    --uniform-mixture-prob "${MIXTURE_PROB}" \
    --uniform-mixture-loss-weight 1.0 \
    --seed 42 \
    --wandb-project "${WANDB_PROJECT}" \
    "$@"
