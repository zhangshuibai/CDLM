#!/usr/bin/env bash
# Open-dCoder-0.5B continued training: CDLM (absorbing + uniform mixture) or MDLM (absorbing only).
#
# Reproduces the paper's headline 0.5B checkpoints, i.e. step 2000 of the long-horizon runs
# run_local_mixture_continue_A100.sh (CDLM) and run_local_mixture_continue_A100_baseline.sh (MDLM),
# and the multi-seed study (seeds 42 / 1234 / 2025).
#
# Learning-rate schedule: --train.max_steps is deliberately NOT passed. veomni derives
#   train_steps  = ceil(train_size / (global_batch_size * max_seq_len)) = ceil(1e12 / (12 * 4096)) = 20,345,053
#   warmup_steps = int(train_steps * lr_warmup_ratio)                    = 20,345
# so step 2000 is still inside the linear warmup (lr = 3e-4 * 2000 / 20345 = 2.949e-5). Passing
# max_steps=2000 would instead give a fully decayed 2000-step cosine, a different model.
# To keep the long horizon and still end at STOP_STEP, save_steps is set to STOP_STEP and this
# wrapper kills the job as soon as the step-STOP_STEP HuggingFace checkpoint has been written.
#
# World size: the published seed-42 checkpoints were trained with NPROC=4 (3 x 4 GPUs x accum 1);
# the additional seeds 1234 and 2025 with NPROC=2 (3 x 2 GPUs x accum 2). The global batch is
# pinned to 12 either way, so the LR schedule is identical; only the per-rank data sharding differs.
#
# Data: DATA_DIR is the Nemotron-SFT-Code/ folder (78 parquet shards) of the Hugging Face dataset
# nvidia/Nemotron-Pretraining-SFT-v1.
#
# Usage:
#   ARM=cdlm SEED=42 bash training/scripts/train_0.5b.sh
#   ARM=mdlm SEED=1234 NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash training/scripts/train_0.5b.sh
#
# Environment variables [default]:
#   ARM                   cdlm | mdlm                                          [cdlm]
#   SEED                  training seed                                        [42]
#   STOP_STEP             optimizer step at which to save and stop             [2000]
#   NPROC                 GPUs on this node: 1, 2 or 4 (global batch stays 12) [4]
#   CUDA_VISIBLE_DEVICES  GPUs to use                                          [0,...,NPROC-1]
#   MASTER_PORT           rendezvous port                                      [29500]
#   DATA_DIR              Nemotron-SFT-Code parquet shards                     [<repo>/data/Nemotron-SFT-Code]
#   TRAIN_PATH            comma-separated shard directories passed verbatim to --data.train_path
#                         instead of DATA_DIR (shard order, see training/data_prep/README.md)
#   OUTPUT_DIR            parent directory of the run directory                [<repo>/training/outputs]
#   RUN_NAME             run directory name        [open-dcoder-0.5B-<ARM>-seed<SEED>-step<STOP_STEP>]
#   BASE_MODEL            initial checkpoint                                   [fredzzp/open-dcoder-0.5B]
#   PRUNE_DCP             1: delete the dcp model/optimizer shards after the HF export [0]
#   POLL_SECONDS          how often the log is checked for the checkpoint      [10]
#   WANDB_MODE            offline | online | disabled                          [offline]
#   WANDB_PROJECT         wandb project (config default if unset); the entity comes from WANDB_ENTITY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/training"

die() { echo "ERROR: $*" >&2; exit 1; }

ARM="${ARM:-cdlm}"
SEED="${SEED:-42}"
STOP_STEP="${STOP_STEP:-2000}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/Nemotron-SFT-Code}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-open-dcoder-0.5B-${ARM}-seed${SEED}-step${STOP_STEP}}"
BASE_MODEL="${BASE_MODEL:-fredzzp/open-dcoder-0.5B}"
PRUNE_DCP="${PRUNE_DCP:-0}"
POLL_SECONDS="${POLL_SECONDS:-10}"

# Mixture Masking hyperparameters
case "${ARM}" in
    cdlm) mixture_prob=0.1; noise_token_wt=0.1; clean_token_wt=0 ;;
    mdlm) mixture_prob=0;   noise_token_wt=0;   clean_token_wt=0 ;;
    *) die "ARM must be cdlm or mdlm (got '${ARM}')" ;;
esac
repr_align_wt=0

[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer (got '${SEED}')"
[[ "${STOP_STEP}" =~ ^[1-9][0-9]*$ ]] || die "STOP_STEP must be a positive integer (got '${STOP_STEP}')"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer (got '${MASTER_PORT}')"
case "${NPROC}" in
    1|2|4) ;;
    *) die "NPROC must be 1, 2 or 4 so that global batch 12 = 3 x NPROC x grad_accum (got '${NPROC}')" ;;
esac

# Batch size configuration
NNODES=1
micro_batch_size=3
global_batch_size=12   # pinned; grad_accum = 12 / (3 * NPROC)
grad_accum=$((global_batch_size / (micro_batch_size * NPROC * NNODES)))

[[ -f "${TRAIN_ROOT}/tasks/train_torch.py" ]] || die "missing ${TRAIN_ROOT}/tasks/train_torch.py"
[[ -f "${TRAIN_ROOT}/configs/pretrain/qwen2_5_coder_500M.yaml" ]] || die "missing ${TRAIN_ROOT}/configs/pretrain/qwen2_5_coder_500M.yaml"
command -v torchrun >/dev/null 2>&1 || die "torchrun not found; activate the training environment first"

# --data.train_path: DATA_DIR, or TRAIN_PATH passed verbatim (comma-separated absolute shard
# directories, e.g. the recorded shard order from training/data_prep/README.md).
order_hint=""
if [[ -n "${TRAIN_PATH:-}" ]]; then
    IFS=',' read -r -a train_path_entries <<< "${TRAIN_PATH}"
    for entry in "${train_path_entries[@]}"; do
        [[ "${entry}" == /* && -e "${entry}" ]] || die "TRAIN_PATH entry is not an existing absolute path: ${entry}"
    done
    train_path="${TRAIN_PATH}"
    train_path_desc="TRAIN_PATH (${#train_path_entries[@]} entries, first ${train_path_entries[0]})"
else
    [[ -d "${DATA_DIR}" ]] || die "DATA_DIR does not exist: ${DATA_DIR} (set DATA_DIR to the Nemotron-SFT-Code parquet directory)"
    DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
    train_path="${DATA_DIR}"
    train_path_desc="${DATA_DIR}"
    order_file="$(dirname "${DATA_DIR}")/train_path_paper_order.txt"
    if [[ -f "${order_file}" ]]; then
        order_hint="NOTE: set TRAIN_PATH=\"\$(cat ${order_file})\" to use the recorded shard order."
    fi
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
# The config enables auto_resume, so a leftover checkpoint would silently be resumed.
if [[ -e "${RUN_DIR}/checkpoints" || -e "${RUN_DIR}/last_checkpoint" ]]; then
    die "${RUN_DIR} already contains checkpoints; remove it or set RUN_NAME"
fi
mkdir -p "${RUN_DIR}"

HF_DIR="${RUN_DIR}/checkpoints/global_step_${STOP_STEP}/hf_ckpt_step_$(printf '%07d' "${STOP_STEP}")"
LOG="${RUN_DIR}/train.log"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
fi
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"

train_args=(
    tasks/train_torch.py
    configs/pretrain/qwen2_5_coder_500M.yaml
    --data.train_path="${train_path}"
    --train.ckpt_manager=dcp
    --train.seed="${SEED}"
    --train.micro_batch_size="${micro_batch_size}"
    --train.global_batch_size="${global_batch_size}"
    --train.output_dir="${RUN_DIR}"
    --train.repr_align_wt="${repr_align_wt}"
    --train.mixture_prob="${mixture_prob}"
    --train.noise_token_wt="${noise_token_wt}"
    --train.clean_token_wt="${clean_token_wt}"
    --train.save_steps="${STOP_STEP}"
    --train.eval_every=0
    --train.eval_before_train=false
    --train.save_time_interval_minutes=0
    --train.wandb_name="${RUN_NAME}"
    --train.use_wandb=true
    --data.datasets_type=iterable
    --model.model_path="${BASE_MODEL}"
    --train.freeze_layers="lm_head,embed_tokens"
)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    train_args+=(--train.wandb_project="${WANDB_PROJECT}")
fi

# Expected schedule for configs/pretrain/qwen2_5_coder_500M.yaml (train_size 1e12, max_seq_len 4096,
# lr 3e-4, lr_warmup_ratio 0.001); the job logs the actual "train_steps" at start-up.
expected_train_steps=$(( (1000000000000 + global_batch_size * 4096 - 1) / (global_batch_size * 4096) ))
expected_warmup=$(( expected_train_steps / 1000 ))
expected_lr=$(awk -v s="${STOP_STEP}" -v w="${expected_warmup}" 'BEGIN { printf "%.6e", 3e-4 * s / w }')

echo "======================================"
echo "Open-dCoder-0.5B continued training"
echo "======================================"
echo "Arm:                ${ARM}"
echo "Seed:               ${SEED}"
echo "Stop step:          ${STOP_STEP} (save_steps=${STOP_STEP}, no max_steps)"
echo "Mixture prob:       ${mixture_prob}"
echo "Noise token weight: ${noise_token_wt}"
echo "Clean token weight: ${clean_token_wt}"
echo "Repr align weight:  ${repr_align_wt}"
echo "Frozen layers:      lm_head,embed_tokens"
echo "Base model:         ${BASE_MODEL}"
echo "GPUs per node:      ${NPROC} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Micro batch size:   ${micro_batch_size}"
echo "Global batch size:  ${global_batch_size} (grad_accum = ${global_batch_size} / (${micro_batch_size} x ${NPROC}) = ${grad_accum})"
echo "LR schedule:        train_steps=${expected_train_steps}, warmup=${expected_warmup}, lr(${STOP_STEP})=${expected_lr} (expected)"
echo "Train path:         ${train_path_desc}"
echo "Run dir:            ${RUN_DIR}"
echo "HF checkpoint:      ${HF_DIR}"
echo "Log:                ${LOG}"
echo "Rendezvous:         localhost:${MASTER_PORT}"
echo "wandb:              WANDB_MODE=${WANDB_MODE}"
echo "======================================"
if [[ "${SEED}" != 42 && "${NPROC}" != 2 ]]; then
    echo "NOTE: the multi-seed runs (seeds 1234 and 2025) were trained with NPROC=2."
fi
if [[ -n "${order_hint}" ]]; then
    echo "${order_hint}"
fi

ere_escape() { printf '%s' "$1" | sed -e 's/[][\.*^$+?(){}|]/\\&/g'; }
WORKER_PATTERN="train\.output_dir=$(ere_escape "${RUN_DIR}")( |$)"
EVAL_PATTERN="pretrained=$(ere_escape "${HF_DIR}"),"

LAUNCHER=""
TAIL_PID=""

# Kill the elastic launcher first, otherwise it tears down or restarts the workers itself;
# then kill every process of this run (workers and dataloader workers carry its output_dir).
stop_training() {
    if [[ -n "${LAUNCHER}" ]]; then
        kill -9 "${LAUNCHER}" 2>/dev/null || true
    fi
    sleep 5
    pkill -9 -f "${WORKER_PATTERN}" 2>/dev/null || true
    pkill -9 -f "${EVAL_PATTERN}" 2>/dev/null || true
    sleep 8
    pkill -9 -f "${WORKER_PATTERN}" 2>/dev/null || true
    sleep 2
}

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "${LAUNCHER}" ]]; then
        echo "Stopping training processes of ${RUN_DIR}" >&2
        stop_training
    fi
    if [[ -n "${TAIL_PID}" ]]; then
        kill "${TAIL_PID}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${TRAIN_ROOT}"
printf '%q ' torchrun --nproc_per_node="${NPROC}" --nnodes="${NNODES}" --rdzv_id="local_mixture_$$" \
    --rdzv_backend=c10d --rdzv_endpoint="localhost:${MASTER_PORT}" "${train_args[@]}" > "${RUN_DIR}/command.sh"
echo >> "${RUN_DIR}/command.sh"

torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes="${NNODES}" \
    --rdzv_id="local_mixture_$$" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${MASTER_PORT}" \
    "${train_args[@]}" > "${LOG}" 2>&1 &
LAUNCHER=$!

tail -n +1 -F --pid="${LAUNCHER}" "${LOG}" 2>/dev/null &
TAIL_PID=$!

HF_MARKER="Huggingface checkpoint saved at ${HF_DIR} successfully"
while true; do
    if grep -qF "${HF_MARKER}" "${LOG}" 2>/dev/null; then
        break
    fi
    if ! kill -0 "${LAUNCHER}" 2>/dev/null; then
        grep -qF "${HF_MARKER}" "${LOG}" 2>/dev/null && break
        die "training exited before the step-${STOP_STEP} checkpoint was written; see ${LOG}"
    fi
    sleep "${POLL_SECONDS}"
done

echo "Step-${STOP_STEP} HF checkpoint written; stopping the run."
stop_training
LAUNCHER=""
kill "${TAIL_PID}" 2>/dev/null || true
TAIL_PID=""

[[ -s "${HF_DIR}/model.safetensors" || -s "${HF_DIR}/model.safetensors.index.json" ]] \
    || die "no weights found in ${HF_DIR}"

# model_assets/ holds config and tokenizer files; make sure the HF directory loads standalone.
for f in "${RUN_DIR}"/model_assets/*; do
    [[ -e "${f}" ]] || continue
    b="$(basename "${f}")"
    [[ -e "${HF_DIR}/${b}" ]] || cp "${f}" "${HF_DIR}/${b}"
done

if [[ "${PRUNE_DCP}" == 1 ]]; then
    rm -rf "${RUN_DIR}/checkpoints/global_step_${STOP_STEP}/model" \
           "${RUN_DIR}/checkpoints/global_step_${STOP_STEP}/optimizer" \
           "${RUN_DIR}/last_checkpoint"
    echo "Removed the dcp model/optimizer shards (PRUNE_DCP=1)."
fi

echo "======================================"
echo "Done: ${HF_DIR}"
echo "Expose it for the evaluation pipeline with:"
echo "  bash ${REPO_ROOT}/training/tools/convert_to_hf.sh ${RUN_DIR}"
echo "======================================"
