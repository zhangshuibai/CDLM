#!/bin/bash
# Run ONE CRB cell (dataset x error_type x n_replace x refined_steps) for one arm,
# pinned to ONE GPU, against the repo's top-level refine_code.py / evaluate_code.py.
#
#   usage: bash run_crb_cell.sh <gpu> <label> <adapter|NONE> <dataset> <error_type> <n_replace> <steps>
#
# Expects the CRB inputs at <repo>/buggy_datasets/<dataset>/evaluated/
# LLaDA-8B-Base_<error_type>_2_wrong_<n_replace>_evaluated.jsonl (see README.md).
# Environment overrides: PYTHON [python].
#
# Outputs go to <repo>/g6v_<label>_results/... and existing outputs are reused. The cell directory
# records a hash of the adapter it was produced with (adapter.txt); if the adapter differs (another
# path, or retrained in place), the cell stops with an error instead of reusing the old outputs.
#
# Numerics are unchanged relative to the 4-GPU launch: --batch_size 1, same
# refine_setting / threshold / temperature, and sharding across GPUs only changes
# which process handles which sample, not how any sample is computed.
set -uo pipefail

GPU=$1; LABEL=$2; ADAPTER=$3; DS=$4; ET=$5; NR=$6; STEPS=$7

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=4

PY=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# A relative adapter path refers to the caller's directory, not to the repo root.
if [ "${ADAPTER}" != "NONE" ] && [ -e "${ADAPTER}" ]; then
    ADAPTER="$(realpath -s -- "${ADAPTER}")"
fi
cd "${REPO}" || exit 1

MODEL_NAME="GSAI-ML/LLaDA-8B-Base"
PREFIX="g6v_${LABEL}"
DATA_NUM=2
ALGO="self_conf-remask:vanilla"
CT=0.9
TEMP=0.0
SETTING="remove_all"

INIT="buggy_datasets/${DS}/evaluated/LLaDA-8B-Base_${ET}_${DATA_NUM}_wrong_${NR}_evaluated.jsonl"
[ -f "${INIT}" ] || { echo "SKIP missing ${INIT}"; exit 0; }

STEM="LLaDA-8B-Base_${ET}_${DATA_NUM}_wrong_${NR}_evaluated"
RDIR="${PREFIX}_results/refined_steps${STEPS}/${SETTING}/self_conf-remask_vanilla_ct090_t00/buggy_datasets/${DS}/evaluated/${STEM}"
RES="${RDIR}/${STEM}_results_refined.jsonl"
STAMP="${RDIR}/adapter.txt"

# Identity of the adapter: NONE, or a hash of the adapter's config and weights. A Hub id
# (<owner>/<name>[/<subfolder>][@<revision>]) is downloaded first and hashed like a local directory,
# so outputs are tied to the weights actually loaded, whatever the Hub revision points to.
ADAPTER_DESC="${ADAPTER}"
if [ "${ADAPTER}" != "NONE" ] && [ ! -d "${ADAPTER}" ]; then
    RESOLVED="$("${PY}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from adapter_path import resolve_adapter; print(resolve_adapter(sys.argv[2]))' \
                "${SCRIPT_DIR}" "${ADAPTER}" | tail -n 1)" \
        || { echo "!!! cannot resolve adapter ${ADAPTER}"; exit 1; }
    ADAPTER_DESC="${ADAPTER} (${RESOLVED})"
    ADAPTER="${RESOLVED}"
fi
if [ "${ADAPTER}" = "NONE" ]; then
    ADAPTER_ID="NONE"
else
    ls "${ADAPTER}"/adapter_config.json "${ADAPTER}"/adapter_model.* >/dev/null 2>&1 \
        || { echo "!!! no adapter_config.json / adapter_model.* in ${ADAPTER}"; exit 1; }
    ADAPTER_ID="sha256:$(cat "${ADAPTER}"/adapter_config.json "${ADAPTER}"/adapter_model.* | sha256sum | cut -c1-16)"
fi
LORA_ARG=()
if [ "${ADAPTER}" != "NONE" ]; then LORA_ARG=(--lora_adapter "${ADAPTER}"); fi

EVAL_SKIP=(--skip_if_exist)
if [ -f "${RES}" ]; then
    OLD_ID="$(head -n 1 "${STAMP}" 2>/dev/null)"
    if [ "${OLD_ID}" != "${ADAPTER_ID}" ]; then
        echo "!!! STALE [${LABEL}] ${RDIR} holds outputs of adapter '${OLD_ID:-unrecorded}', not '${ADAPTER_ID}' (${ADAPTER_DESC}); remove that directory or use another label"
        exit 1
    fi
else
    # Fresh refinement: record the adapter and do not reuse evaluations of an earlier result file.
    if ! { mkdir -p "${RDIR}" && printf '%s\n%s\n' "${ADAPTER_ID}" "${ADAPTER_DESC}" > "${STAMP}"; }; then
        echo "!!! cannot write ${STAMP}"; exit 1
    fi
    EVAL_SKIP=()
fi

echo "### [${LABEL}] gpu${GPU} ${DS}/${ET}/n${NR} steps=${STEPS} adapter=${ADAPTER_ID} START $(date +%H:%M:%S)"

"${PY}" "${SCRIPT_DIR}/refine_code_lora.py" \
    ${LORA_ARG[@]+"${LORA_ARG[@]}"} \
    --initial_results_file "${INIT}" \
    --model_name "${MODEL_NAME}" \
    --batch_size 1 \
    --refined_steps "${STEPS}" \
    --algorithm "${ALGO}" \
    --temperature "${TEMP}" \
    --refine_setting "${SETTING}" \
    --confidence_threshold "${CT}" \
    --output_prefix "${PREFIX}" \
    --skip_existing \
  || { echo "!!! REFINE FAILED [${LABEL}] ${DS}/${ET}/n${NR}/s${STEPS}"; exit 1; }

[ -f "${RES}" ] || { echo "!!! NO RESULT ${RES}"; exit 1; }

# ---- raw evaluation (paper protocol: --no_postprocess) ----
"${PY}" "${REPO}/evaluate_code.py" \
    --results_file "${RES}" \
    --output_file "${RDIR}/${STEM}_results_refined_evaluated.jsonl" \
    --dataset "${DS}" --no_postprocess \
    --summary_file "${RDIR}/pass_at_1_summary.json" \
    --summary_metadata "label:${LABEL},dataset:${DS},error_type:${ET},n_replace:${NR},refined_steps:${STEPS},algorithm:${ALGO},confidence_threshold:${CT},temperature:${TEMP},refine_setting:${SETTING},adapter:${ADAPTER_DESC}" \
    ${EVAL_SKIP[@]+"${EVAL_SKIP[@]}"} \
  || { echo "!!! EVAL FAILED [${LABEL}] ${DS}/${ET}/n${NR}/s${STEPS}"; exit 1; }

# ---- de-fenced evaluation (identical rule for every arm; see defence.py) ----
DEF="${RDIR}/${STEM}_results_refined_defenced.jsonl"
"${PY}" "${SCRIPT_DIR}/defence.py" --in_file "${RES}" --out_file "${DEF}" \
  || { echo "!!! DEFENCE FAILED [${LABEL}] ${DS}/${ET}/n${NR}/s${STEPS}"; exit 1; }
"${PY}" "${REPO}/evaluate_code.py" \
    --results_file "${DEF}" \
    --output_file "${RDIR}/${STEM}_results_refined_defenced_evaluated.jsonl" \
    --dataset "${DS}" --no_postprocess \
    --summary_file "${RDIR}/pass_at_1_summary_defenced.json" \
    --summary_metadata "label:${LABEL},dataset:${DS},error_type:${ET},n_replace:${NR},refined_steps:${STEPS},algorithm:${ALGO},confidence_threshold:${CT},temperature:${TEMP},refine_setting:${SETTING},adapter:${ADAPTER_DESC}" \
    ${EVAL_SKIP[@]+"${EVAL_SKIP[@]}"} \
  || { echo "!!! EVAL FAILED (de-fenced) [${LABEL}] ${DS}/${ET}/n${NR}/s${STEPS}"; exit 1; }

echo "### [${LABEL}] gpu${GPU} ${DS}/${ET}/n${NR} steps=${STEPS} DONE $(date +%H:%M:%S)"
