#!/usr/bin/env bash
# Expose a trained Open-dCoder-0.5B checkpoint as a HuggingFace directory for the evaluation
# pipeline (top-level refine_code.py / utils.py).
#
# utils.load_model_and_tokenizer loads the diffusion Qwen2ForCausalLM from veomni only when the
# model path contains "open-dcoder", so the model is exposed as $MODELS_DIR/<NAME> with a NAME
# that contains "open-dcoder".
#
# If training already exported the requested step (hf_ckpt_step_<7-digit step> from train_0.5b.sh,
# hf_ckpt from the max_steps runs), that directory is linked or copied. Otherwise the dcp model
# shards of the checkpoint are converted with convert_dcp_to_hf.py.
#
# Usage:
#   bash training/tools/convert_to_hf.sh RUN_OR_CKPT_DIR [NAME]
#   STEP=2 bash training/tools/convert_to_hf.sh RUN_DIR      # a train_0.5b.sh run with STOP_STEP=2
#     RUN_OR_CKPT_DIR  a run directory (with checkpoints/ and model_assets/), or a single checkpoint
#                      directory such as <run>/checkpoints/global_step_2000
#     NAME             exported name; "open-dcoder-0.5B-" is prepended unless it already contains
#                      "open-dcoder"                                   [basename of the run directory]
#
# Environment variables [default]:
#   STEP           step to export; for a run directory, set it to the run's STOP_STEP when that is
#                  not 2000 (train_0.5b.sh prints the command with it)
#                                                          [2000, or N for .../global_step_N]
#   MODELS_DIR     where the model is exposed              [<repo>/training/outputs/models]
#   MODE           link | copy                             [link]
#   FORCE_CONVERT  1: convert the dcp shards even if training wrote an HF export [0]
#   PYTHON         interpreter of the training environment [python]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/training"

die() { echo "ERROR: $*" >&2; exit 1; }
has_weights() { [[ -s "$1/model.safetensors" || -s "$1/model.safetensors.index.json" ]]; }

[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 RUN_OR_CKPT_DIR [NAME]"
[[ -d "$1" ]] || die "not a directory: $1"
SRC="$(cd "$1" && pwd)"

MODELS_DIR="${MODELS_DIR:-${TRAIN_ROOT}/outputs/models}"
MODE="${MODE:-link}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
PYTHON="${PYTHON:-python}"
case "${MODE}" in
    link|copy) ;;
    *) die "MODE must be link or copy (got '${MODE}')" ;;
esac

if [[ -d "${SRC}/checkpoints" ]]; then
    RUN_DIR="${SRC}"
    STEP="${STEP:-2000}"
    CKPT_DIR="${RUN_DIR}/checkpoints/global_step_${STEP}"
else
    CKPT_DIR="${SRC}"
    RUN_DIR="$(dirname "$(dirname "${CKPT_DIR}")")"
    if [[ "$(basename "${CKPT_DIR}")" =~ ^global_step_([0-9]+)$ ]]; then
        dir_step="${BASH_REMATCH[1]}"
        [[ -z "${STEP:-}" || "${STEP}" == "${dir_step}" ]] \
            || die "STEP=${STEP} does not match ${CKPT_DIR}"
        STEP="${dir_step}"
    else
        STEP="${STEP:-2000}"
    fi
fi
[[ "${STEP}" =~ ^[0-9]+$ ]] || die "STEP must be an integer (got '${STEP}')"
if [[ ! -d "${CKPT_DIR}" ]]; then
    steps=""
    for d in "${RUN_DIR}"/checkpoints/global_step_*; do
        if [[ -d "${d}" ]]; then steps+=" ${d##*_}"; fi
    done
    die "checkpoint directory not found: ${CKPT_DIR}${steps:+ (steps in this run:${steps}; set STEP=N)}"
fi
ASSETS_DIR="${RUN_DIR}/model_assets"

NAME="${2:-$(basename "${RUN_DIR}")}"
[[ "${NAME}" != */* && -n "${NAME}" ]] || die "NAME must be a plain directory name (got '${NAME}')"
if [[ "${NAME,,}" != *open-dcoder* ]]; then
    NAME="open-dcoder-0.5B-${NAME}"
fi
mkdir -p "${MODELS_DIR}"
MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)"
TARGET="${MODELS_DIR}/${NAME}"

# An HF export written during training for exactly this step, if any.
HF_SRC=""
if [[ "${FORCE_CONVERT}" != 1 ]]; then
    candidates=("${CKPT_DIR}/hf_ckpt_step_$(printf '%07d' "${STEP}")")
    if [[ "$(basename "${CKPT_DIR}")" == global_step_* ]]; then
        candidates+=("${CKPT_DIR}/hf_ckpt")
    fi
    for cand in "${candidates[@]}"; do
        if has_weights "${cand}"; then
            HF_SRC="${cand}"
            break
        fi
    done
fi

echo "======================================"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Step:           ${STEP}"
echo "Model assets:   ${ASSETS_DIR}"
echo "Target:         ${TARGET}"
if [[ -n "${HF_SRC}" ]]; then
    echo "Source:         ${HF_SRC} (HF export written during training, MODE=${MODE})"
else
    echo "Source:         ${CKPT_DIR}/model (dcp shards, converted with convert_dcp_to_hf.py)"
fi
echo "======================================"

if [[ -e "${TARGET}" && ! -L "${TARGET}" ]]; then
    die "${TARGET} already exists; remove it or pass another NAME"
fi

if [[ -n "${HF_SRC}" ]]; then
    # model_assets/ holds config and tokenizer files; make sure the HF directory loads standalone.
    if [[ -d "${ASSETS_DIR}" ]]; then
        for f in "${ASSETS_DIR}"/*; do
            [[ -f "${f}" ]] || continue
            b="$(basename "${f}")"
            [[ -e "${HF_SRC}/${b}" ]] || cp "${f}" "${HF_SRC}/${b}"
        done
    fi
    if [[ "${MODE}" == link ]]; then
        ln -sfn "${HF_SRC}" "${TARGET}"
    else
        [[ ! -L "${TARGET}" ]] || rm -f "${TARGET}"
        cp -r "${HF_SRC}" "${TARGET}"
    fi
else
    [[ -d "${CKPT_DIR}/model" ]] \
        || die "no HF export for step ${STEP} and no dcp model shards in ${CKPT_DIR}/model"
    [[ -f "${ASSETS_DIR}/config.json" ]] || die "missing ${ASSETS_DIR}/config.json"
    [[ ! -L "${TARGET}" ]] || rm -f "${TARGET}"
    tmp_dir="${TARGET}.tmp.$$"
    trap 'rm -rf "${tmp_dir}"' EXIT
    PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="" \
        "${PYTHON}" "${SCRIPT_DIR}/convert_dcp_to_hf.py" \
        --ckpt-dir "${CKPT_DIR}" \
        --model-assets-dir "${ASSETS_DIR}" \
        --save-dir "${tmp_dir}" \
        --expect-step "${STEP}"
    mv "${tmp_dir}" "${TARGET}"
    trap - EXIT
fi

has_weights "${TARGET}" || die "no weights in ${TARGET}"
[[ -f "${TARGET}/config.json" ]] || die "no config.json in ${TARGET}"
[[ -f "${TARGET}/tokenizer_config.json" ]] || die "no tokenizer files in ${TARGET}"

echo "Model ready: ${TARGET}"
echo "Pass it to the evaluation pipeline as --model_name ${TARGET}"
