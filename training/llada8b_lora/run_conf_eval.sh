#!/bin/bash
# G6: confidence-gap + Top-K hit-rate evaluation for base / MDLM / CDLM.
#   usage: bash run_conf_eval.sh <out_dir> <mdlm_adapter> <cdlm_adapter> [extra args...]
# Environment overrides: CUDA_VISIBLE_DEVICES [0], PYTHON [python].
set -uo pipefail

OUT=$1; MDLM=$2; CDLM=$3; shift 3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUT}"

"${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label base \
    --out "${OUT}/conf_base.json" "$@"
"${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label mdlm --adapter "${MDLM}" \
    --out "${OUT}/conf_mdlm.json" "$@"
"${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label cdlm --adapter "${CDLM}" \
    --out "${OUT}/conf_cdlm.json" "$@"
