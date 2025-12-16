#!/bin/bash
set -euo pipefail
source ~/.bashrc
conda activate test-dllm
cd Open-dLLM/eval/eval_completion
# Define model paths as an array (add more models as needed)
MODEL_PATHS=(
    "fredzzp/open-dcoder-0.5B"
    "Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000"
    "Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000"
    # "Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step10k-H200"
)

MAX_NEW_TOKENS=128
STEPS=128
TEMPERATURE=0.8
# ALG="p2_upgraded"
ALG="p2_upgraded_ReMDM"
NUM_PROCESSES=4
WANDB_PROJECT_NAME="dlm-agents-eval"  # Set your wandb project name here, e.g., "dllm-eval"


export CUDA_VISIBLE_DEVICES="0,1,3,5"
export HF_ALLOW_CODE_EVAL=1

# Increase NCCL timeout to prevent timeout errors (set to 2 hours)
export NCCL_TIMEOUT=7200
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1

# Function to extract model name from path for output directory
get_model_name() {
    local path=$1
    # Replace '/' with '-' and remove common prefixes
    echo "$path" | sed 's/\//-/g' | sed 's/^fredzzp-//'
}

# Loop through each model
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(get_model_name "$MODEL_PATH")
    echo "=========================================="
    echo "Evaluating model: $MODEL_PATH"
    echo "Model name: $MODEL_NAME"
    echo "=========================================="
    
    # accelerate launch --num_processes $NUM_PROCESSES eval.py \
    #     --model custom_coder \
    #     --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    #     --tasks humaneval \
    #     --num_fewshot 0 \
    #     --batch_size 5 \
    #     --output_path evals_results/${MODEL_NAME}/humaneval-ns0 \
    #     --log_samples \
    #     --confirm_run_unsafe_code \
    #     ${WANDB_PROJECT_NAME:+--wandb_project_name $WANDB_PROJECT_NAME}

    accelerate launch --num_processes $NUM_PROCESSES eval.py \
        --model custom_coder \
        --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
        --tasks humaneval_plus \
        --num_fewshot 0 \
        --batch_size 5 \
        --output_path evals_results/${MODEL_NAME}/humaneval_plus-ns0 \
        --log_samples \
        --confirm_run_unsafe_code \
        ${WANDB_PROJECT_NAME:+--wandb_project_name $WANDB_PROJECT_NAME}

    accelerate launch --num_processes $NUM_PROCESSES eval.py \
        --model custom_coder \
        --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
        --tasks mbpp \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path evals_results/${MODEL_NAME}/mbpp-ns0 \
        --log_samples \
        --confirm_run_unsafe_code \
        ${WANDB_PROJECT_NAME:+--wandb_project_name $WANDB_PROJECT_NAME}

    accelerate launch --num_processes $NUM_PROCESSES eval.py \
        --model custom_coder \
        --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
        --tasks mbpp_plus \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path evals_results/${MODEL_NAME}/mbpp_plus-ns0 \
        --log_samples \
        --confirm_run_unsafe_code \
        ${WANDB_PROJECT_NAME:+--wandb_project_name $WANDB_PROJECT_NAME}
    
    echo "Completed evaluation for model: $MODEL_PATH"
    echo ""
done

echo "All model evaluations completed!"

