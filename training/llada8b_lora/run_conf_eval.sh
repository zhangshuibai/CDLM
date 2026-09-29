#!/bin/bash
# G6: confidence-gap + Top-K hit-rate evaluation for base / MDLM / CDLM.
#   usage: bash run_conf_eval.sh <out_dir> <mdlm_adapter> <cdlm_adapter> [extra args...]
# An adapter is a local adapter directory or a Hub id <owner>/<name>[@<revision>].
# Environment overrides: CUDA_VISIBLE_DEVICES [0], PYTHON [python],
#   LABEL_SUFFIX [""]  appended to the mdlm / cdlm labels and file names, e.g. _oci gives
#                      conf_mdlm_oci.json and conf_cdlm_oci.json
#   SKIP_BASE    [0]   set to 1 to skip the base model (conf_base.json)
set -uo pipefail

OUT=$1; MDLM=$2; CDLM=$3; shift 3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFX=${LABEL_SUFFIX:-}
mkdir -p "${OUT}"

if [ "${SKIP_BASE:-0}" != 1 ]; then
    "${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label base \
        --out "${OUT}/conf_base.json" "$@"
fi
"${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label "mdlm${SFX}" --adapter "${MDLM}" \
    --out "${OUT}/conf_mdlm${SFX}.json" "$@"
"${PY}" "${SCRIPT_DIR}/eval_confidence.py" --label "cdlm${SFX}" --adapter "${CDLM}" \
    --out "${OUT}/conf_cdlm${SFX}.json" "$@"
