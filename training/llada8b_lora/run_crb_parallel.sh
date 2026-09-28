#!/bin/bash
# Drive the whole CRB sweep with one concurrent single-GPU cell queue per GPU.
#   usage: bash run_crb_parallel.sh <n_replace>
#
# Environment overrides (defaults in brackets):
#   GPUS         ["0 1 2 3"]
#   OUTPUT_DIR   [<repo>/outputs/llada8b_lora]  (same default as run_train.sh)
#   MDLM_ADAPTER [${OUTPUT_DIR}/full_mdlm/final]
#   CDLM_ADAPTER [${OUTPUT_DIR}/full_cdlm/final]
#   LOG_DIR      [${OUTPUT_DIR}]                 per-GPU logs crb_gpu<N>.log
#
# Every cell is independent, so this is scheduling-only: identical numerics to the
# previous `torchrun --nproc_per_node=4` launch (batch_size 1 either way), just one
# cell per GPU instead of one cell spread over four.
set -uo pipefail

NR=${1:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"

OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/llada8b_lora}
MDLM=${MDLM_ADAPTER:-${OUTPUT_DIR}/full_mdlm/final}
CDLM=${CDLM_ADAPTER:-${OUTPUT_DIR}/full_cdlm/final}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}}
mkdir -p "${LOG_DIR}"

# Build the full cell list: arm x dataset x error_type x steps
CELLS=()
for ARM in "base:NONE" "mdlm:${MDLM}" "cdlm:${CDLM}"; do
    LABEL="${ARM%%:*}"; ADAPTER="${ARM#*:}"
    for DS in human-eval human-eval+ mbpp mbpp+; do
        for ET in operator var literal; do
            for STEPS in 2 5; do
                CELLS+=("${LABEL}|${ADAPTER}|${DS}|${ET}|${STEPS}")
            done
        done
    done
done

echo "total cells: ${#CELLS[@]}  (n_replace=${NR}), ${#GPUS[@]}-way parallel over GPUs ${GPUS[*]}"

# Round-robin: slot i takes cells i, i+G, i+2G, ... so each GPU runs a serial queue.
for i in "${!GPUS[@]}"; do
    (
        GPU=${GPUS[$i]}
        for ((j=i; j<${#CELLS[@]}; j+=${#GPUS[@]})); do
            IFS='|' read -r LABEL ADAPTER DS ET STEPS <<< "${CELLS[$j]}"
            bash "${SCRIPT_DIR}/run_crb_cell.sh" "${GPU}" "${LABEL}" "${ADAPTER}" \
                 "${DS}" "${ET}" "${NR}" "${STEPS}" \
                 >> "${LOG_DIR}/crb_gpu${GPU}.log" 2>&1
        done
        echo "queue for GPU ${GPU} finished"
    ) &
done
wait
echo "=== ALL CELLS COMPLETE (n_replace=${NR}) ==="
