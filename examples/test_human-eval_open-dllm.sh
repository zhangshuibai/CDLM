#!/bin/bash
# Exit on error, undefined variables, and pipe failures
set -euo pipefail
source ~/.bashrc
conda activate test-dllm
export CUDA_VISIBLE_DEVICES=7

# Configuration
MODEL_NAME="fredzzp/open-dcoder-0.5B"
MODEL_DISPLAY_NAME=$(basename "$MODEL_NAME")
DATASET="human-eval"
ERROR_TYPE="operator"
N_REPLACE=1
DATA_NUM=2
REFINED_STEPS=2
TEMPERATURE=0.0
ALGORITHM="self_conf-remask:vanilla"
CONFIDENCE_THRESHOLD=0.90
REFINE_SETTING="remove_all"
DATA_PATH_BASE="buggy_datasets"
MASTER_PORT=29502

echo "=========================================="
echo "Testing LLaDA-8B-Base on HumanEval"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Error type: $ERROR_TYPE"
echo "N_REPLACE: $N_REPLACE"
echo "Refined steps: $REFINED_STEPS"
echo "Temperature: $TEMPERATURE"
echo "=========================================="

# (1) Generate buggy data
BUGGY_DIR="${DATA_PATH_BASE}/${DATASET}"
mkdir -p "${BUGGY_DIR}"
BUGGY_DATA_FILE="${BUGGY_DIR}/${MODEL_DISPLAY_NAME}_${ERROR_TYPE}_${DATA_NUM}_wrong_${N_REPLACE}.jsonl"

echo "Generating buggy data: ${BUGGY_DATA_FILE}"
python codecorrection/generate.py \
    --dataset "${DATASET}" \
    --error_type "${ERROR_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_PATH_BASE}" \
    --n_replace "${N_REPLACE}" \
    --data_num "${DATA_NUM}" \
    --deduplicate

# (2) Evaluate buggy data
EVALUATED_DIR="${BUGGY_DIR}/evaluated"
mkdir -p "${EVALUATED_DIR}"
INITIAL_RESULTS_FILE="${EVALUATED_DIR}/${MODEL_DISPLAY_NAME}_${ERROR_TYPE}_${DATA_NUM}_wrong_${N_REPLACE}_evaluated.jsonl"

echo "Evaluating buggy data: ${INITIAL_RESULTS_FILE}"
python evaluate_code.py \
    --results_file "${BUGGY_DATA_FILE}" \
    --output_file "${INITIAL_RESULTS_FILE}" \
    --dataset "${DATASET}" \
    --map_prompt2completion \
    --no_postprocess

# (3) Run refine_code.py
echo "Running refinement..."

# Build algorithm suffix with parameters (matching build_output_paths logic)
ALGORITHM_SAFE=$(echo "$ALGORITHM" | sed 's/:/_/g')
ALGORITHM_SUFFIX="${ALGORITHM_SAFE}"
if [ "$ALGORITHM" = "self_conf-remask:vanilla" ] && [ -n "${CONFIDENCE_THRESHOLD}" ]; then
    CT_STR=$(echo "${CONFIDENCE_THRESHOLD}" | sed 's/\.//')
    ALGORITHM_SUFFIX="${ALGORITHM_SAFE}_ct${CT_STR}"
fi
if [ -n "${TEMPERATURE}" ]; then
    TEMP_STR=$(echo "${TEMPERATURE}" | sed 's/\.//')
    ALGORITHM_SUFFIX="${ALGORITHM_SUFFIX}_t${TEMP_STR}"
fi

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} refine_code.py \
    --initial_results_file "${INITIAL_RESULTS_FILE}" \
    --model_name "${MODEL_NAME}" \
    --batch_size 1 \
    --refined_steps "${REFINED_STEPS}" \
    --algorithm "${ALGORITHM}" \
    --temperature "${TEMPERATURE}" \
    --refine_setting "${REFINE_SETTING}" \
    --confidence_threshold "${CONFIDENCE_THRESHOLD}"

wait

# (4) Extract output paths from refine_code.py logic (matching build_output_paths)
INPUT_STEM=$(basename "${INITIAL_RESULTS_FILE}" .jsonl)
INPUT_DIR=$(dirname "${INITIAL_RESULTS_FILE}")
REFINED_RESULTS_DIR="correction_results/refined_steps${REFINED_STEPS}/${REFINE_SETTING}/${ALGORITHM_SUFFIX}/${INPUT_DIR}/${INPUT_STEM}"
REFINED_RESULTS_FILE="${REFINED_RESULTS_DIR}/${INPUT_STEM}_results_refined.jsonl"
REFINED_HISTORY_DIR="correction_history/refined_steps${REFINED_STEPS}/${REFINE_SETTING}/${ALGORITHM_SUFFIX}/${INPUT_DIR}/${INPUT_STEM}"

# (5) Evaluate refined results
REFINED_EVALUATED_FILE="${REFINED_RESULTS_DIR}/${INPUT_STEM}_results_refined_evaluated.jsonl"
SUMMARY_FILE="${REFINED_RESULTS_DIR}/pass_at_1_summary.json"
if [ -f "${REFINED_RESULTS_FILE}" ]; then
    echo "Evaluating refined results: ${REFINED_RESULTS_FILE}"
    python evaluate_code.py \
        --results_file "${REFINED_RESULTS_FILE}" \
        --output_file "${REFINED_EVALUATED_FILE}" \
        --dataset "${DATASET}" \
        --no_postprocess \
        --summary_file "${SUMMARY_FILE}" \
        --summary_metadata "dataset:${DATASET},error_type:${ERROR_TYPE},n_replace:${N_REPLACE},model_name:${MODEL_NAME},data_num:${DATA_NUM},refined_steps:${REFINED_STEPS},algorithm:${ALGORITHM},confidence_threshold:${CONFIDENCE_THRESHOLD},temperature:${TEMPERATURE},refine_setting:${REFINE_SETTING}"
    echo "Refined results evaluation saved to: ${REFINED_EVALUATED_FILE}"
else
    echo "Warning: ${REFINED_RESULTS_FILE} does not exist, skipping evaluation"
fi

echo "=========================================="
echo "Completed testing!"
echo "Refined results: ${REFINED_RESULTS_FILE}"
if [ -f "${REFINED_EVALUATED_FILE:-}" ]; then
    echo "Refined evaluation: ${REFINED_EVALUATED_FILE}"
fi
echo "=========================================="

