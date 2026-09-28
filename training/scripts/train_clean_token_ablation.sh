#!/usr/bin/env bash
# Clean-token supervision ablation for Open-dCoder-0.5B (paper: CDLM + ctw, "Effect of supervising
# clean tokens"): mixture_prob=0.1, noise_token_wt=0.1, clean_token_wt=0.1.
#
# Mirrors mixture_H200_ablation_mp0.1_ntw_0.1_ctw_0.1.sh, including --train.max_steps=2000, so the
# LR schedule is a complete 2000-step cosine (2 warmup steps, decaying to lr_min=3e-6), like the
# alpha sweep and unlike the long-horizon headline runs in train_0.5b.sh.
#
# eval_before_train / eval_every=1000 are kept from the original script. They save extra
# checkpoints under checkpoints/eval_before_train and checkpoints/eval, and rank 0 then runs
# eval/eval_completion/eval_single.py, which is not shipped here; train_torch.py ignores the
# failure, so training is unaffected.
#
# Data: DATA_DIR is the Nemotron-SFT-Code/ folder (78 parquet shards) of the Hugging Face dataset
# nvidia/Nemotron-Pretraining-SFT-v1. If data_prep/prepare_nemotron_sft_code.py --paper_order has
# written DATA_DIR/../train_path_paper_order.txt, the shards are read in that recorded order.
#
# Usage:
#   bash training/scripts/train_clean_token_ablation.sh
#
# The HF checkpoint is written to
#   $OUTPUT_DIR/mixture-mdm-mp0.1-ntw0.1-ctw0.1-<timestamp>/checkpoints/global_step_2000/hf_ckpt
#
# Environment variables [default]:
#   NPROC                 GPUs on this node: 1, 2 or 4 (global batch stays 12) [4]
#   CUDA_VISIBLE_DEVICES  GPUs to use                                          [0,...,NPROC-1]
#   MASTER_PORT           rendezvous port                                      [29400]
#   DATA_DIR              Nemotron-SFT-Code parquet shards                     [<repo>/data/Nemotron-SFT-Code]
#   PAPER_ORDER           1: use DATA_DIR/../train_path_paper_order.txt when it exists;
#                         0: os.listdir order of DATA_DIR                      [1]
#   TRAIN_PATH            comma-separated shard directories passed verbatim to --data.train_path
#                         instead of DATA_DIR (shard order, see training/data_prep/README.md)
#   OUTPUT_DIR            parent directory of the run directory                [<repo>/training/outputs]
#   BASE_MODEL            initial checkpoint (Hub id or local directory)       [fredzzp/open-dcoder-0.5B]
#   WANDB_MODE            offline | online | disabled                          [offline]
#   WANDB_PROJECT         wandb project                                        [mixture_ablation]
#   WANDB_ENTITY          wandb entity                                         [unset: the W&B default entity]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/training"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -eq 0 ]] || die "usage: $0   (no arguments; configure with environment variables)"

NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29400}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/Nemotron-SFT-Code}"
PAPER_ORDER="${PAPER_ORDER:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}/outputs}"
BASE_MODEL="${BASE_MODEL:-fredzzp/open-dcoder-0.5B}"
WANDB_PROJECT="${WANDB_PROJECT:-mixture_ablation}"

[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer (got '${MASTER_PORT}')"
[[ "${PAPER_ORDER}" == 0 || "${PAPER_ORDER}" == 1 ]] || die "PAPER_ORDER must be 0 or 1 (got '${PAPER_ORDER}')"
case "${NPROC}" in
    1|2|4) ;;
    *) die "NPROC must be 1, 2 or 4 so that global batch 12 = 3 x NPROC x grad_accum (got '${NPROC}')" ;;
esac
# A local model directory is resolved here, before the cd into training/.
if [[ -e "${BASE_MODEL}" ]]; then
    BASE_MODEL="$(realpath -s -- "${BASE_MODEL}")"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
fi
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

# Ensure we use the veomni module shipped in training/
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES=1
RDV_ADDR=localhost

# Batch size configuration
micro_batch_size=3
global_batch_size=12   # = 3 x 4 GPUs in the original run; grad_accum = 12 / (3 * NPROC)
grad_accum=$((global_batch_size / (micro_batch_size * NPROC * NNODES)))

# Mixture Masking hyperparameters
mixture_prob=0.1       # 10% of non-masked tokens become noise
noise_token_wt=0.1     # Weight for noise token loss
clean_token_wt=0.1     # Weight for clean token loss
repr_align_wt=0        # Knowledge distillation weight

# Evaluation configuration
eval_batch_size=10     # Batch size for the in-training HumanEval hook

[[ -f "${TRAIN_ROOT}/tasks/train_torch.py" ]] || die "missing ${TRAIN_ROOT}/tasks/train_torch.py"
[[ -f "${TRAIN_ROOT}/configs/pretrain/qwen2_5_coder_500M.yaml" ]] || die "missing ${TRAIN_ROOT}/configs/pretrain/qwen2_5_coder_500M.yaml"
command -v torchrun >/dev/null 2>&1 || die "torchrun not found; activate the training environment first"

# Prints the first line of $1 (the comma-separated shard directories written by
# prepare_nemotron_sft_code.py --paper_order) if those directories hold exactly the files of $2.
paper_order_path() (
    shopt -s nullglob dotglob
    IFS= read -r line < "$1" || [[ -n "${line:-}" ]] || exit 1
    IFS=',' read -r -a dirs <<< "${line}"
    files=()
    for d in "${dirs[@]}"; do
        [[ "${d}" == /* && -d "${d}" ]] || exit 1
        files+=("${d}"/*)
    done
    data=("$2"/*)
    (( ${#files[@]} > 0 && ${#files[@]} == ${#data[@]} )) || exit 1
    [[ "$(stat -L -c '%d:%i' -- "${files[@]}" 2>/dev/null | sort)" \
        == "$(stat -L -c '%d:%i' -- "${data[@]}" 2>/dev/null | sort)" ]] || exit 1
    printf '%s\n' "${line}"
)

# --data.train_path: TRAIN_PATH passed verbatim (comma-separated absolute shard directories), else
# the recorded shard order next to DATA_DIR if present, else DATA_DIR itself.
if [[ -n "${TRAIN_PATH:-}" ]]; then
    IFS=',' read -r -a train_path_entries <<< "${TRAIN_PATH}"
    for entry in "${train_path_entries[@]}"; do
        [[ "${entry}" == /* && -e "${entry}" ]] || die "TRAIN_PATH entry is not an existing absolute path: ${entry}"
    done
    train_path="${TRAIN_PATH}"
    train_path_desc="TRAIN_PATH (${#train_path_entries[@]} entries, first ${train_path_entries[0]})"
    shard_order="as listed in TRAIN_PATH"
else
    [[ -d "${DATA_DIR}" ]] || die "DATA_DIR does not exist: ${DATA_DIR} (set DATA_DIR to the Nemotron-SFT-Code parquet directory)"
    DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
    train_path="${DATA_DIR}"
    train_path_desc="${DATA_DIR}"
    order_file="$(dirname "${DATA_DIR}")/train_path_paper_order.txt"
    shard_order="os.listdir order of ${DATA_DIR} (filesystem dependent, not the paper order)"
    if [[ ! -f "${order_file}" ]]; then
        shard_order+="; no ${order_file}"
    elif [[ "${PAPER_ORDER}" == 0 ]]; then
        shard_order+="; PAPER_ORDER=0"
    elif train_path="$(paper_order_path "${order_file}" "${DATA_DIR}")"; then
        train_path_desc="shards of ${DATA_DIR} via ${order_file}"
        shard_order="paper order from ${order_file}"
    else
        die "${order_file} does not list exactly the shards of ${DATA_DIR}; regenerate it with
       python training/data_prep/prepare_nemotron_sft_code.py --local_dir $(dirname "${DATA_DIR}") --skip_download --paper_order
       or set PAPER_ORDER=0 to use the os.listdir order"
    fi
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

# Experiment configuration
exp_name=mixture-mdm-mp${mixture_prob}-ntw${noise_token_wt}-ctw${clean_token_wt}-$(date +%Y%m%d-%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${exp_name}"
mkdir -p "${RUN_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"

echo "======================================"
echo "Running dLLM Mixture Masking Training (clean-token ablation)"
echo "======================================"
echo "Experiment: ${exp_name}"
echo "Output Dir: ${RUN_DIR}"
echo "Train path: ${train_path_desc}"
echo "Shard order: ${shard_order}"
echo "Base model: ${BASE_MODEL}"
echo "GPUs per node: ${NPROC} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Nodes: ${NNODES}"
echo "Micro batch size: ${micro_batch_size}"
echo "Global batch size: ${global_batch_size} (grad_accum ${grad_accum})"
echo "Max steps: 2000 (full cosine schedule within 2000 steps)"
echo "Mixture prob: ${mixture_prob}"
echo "Noise token weight: ${noise_token_wt}"
echo "Clean token weight: ${clean_token_wt}"
echo "Repr align weight: ${repr_align_wt}"
echo "Eval batch size: ${eval_batch_size}"
echo "Rendezvous: ${RDV_ADDR}:${MASTER_PORT}"
echo "wandb: WANDB_MODE=${WANDB_MODE}, project=${WANDB_PROJECT}, entity=${WANDB_ENTITY:-<W&B default>}"
echo "======================================"

# --train.wandb_entity is always passed: the config's "wandb_entity: null" would otherwise reach
# wandb as the entity "null" and override WANDB_ENTITY; an empty entity means the W&B default.
cd "${TRAIN_ROOT}"
torchrun \
   --nproc_per_node="${NPROC}" \
   --nnodes="${NNODES}" \
   --rdzv_id=local_mixture_$$  \
   --rdzv_backend=c10d \
   --rdzv_endpoint="${RDV_ADDR}:${MASTER_PORT}" \
    tasks/train_torch.py \
    configs/pretrain/qwen2_5_coder_500M.yaml \
    --data.train_path="${train_path}" \
    --train.ckpt_manager=dcp \
    --train.micro_batch_size="${micro_batch_size}" \
    --train.global_batch_size="${global_batch_size}" \
    --train.output_dir="${RUN_DIR}" \
    --train.repr_align_wt="${repr_align_wt}" \
    --train.mixture_prob="${mixture_prob}" \
    --train.noise_token_wt="${noise_token_wt}" \
    --train.clean_token_wt="${clean_token_wt}" \
    --train.max_steps=2000 \
    --train.num_train_epochs=1 \
    --train.save_steps=10000 \
    --train.eval_every=1000 \
    --train.eval_batch_size="${eval_batch_size}" \
    --train.wandb_name="${exp_name}" \
    --train.use_wandb=true \
    --data.datasets_type=iterable \
    --model.model_path="${BASE_MODEL}" \
    --train.freeze_layers="lm_head,embed_tokens" \
    --train.eval_before_train=true \
    --train.wandb_project="${WANDB_PROJECT}" \
    --train.wandb_entity="${WANDB_ENTITY:-}"

HF_DIR="${RUN_DIR}/checkpoints/global_step_2000/hf_ckpt"
[[ -s "${HF_DIR}/model.safetensors" || -s "${HF_DIR}/model.safetensors.index.json" ]] \
    || die "training finished but no weights were found in ${HF_DIR}"
echo "======================================"
echo "Done: ${HF_DIR}"
echo "Expose it for the evaluation pipeline with:"
echo "  bash ${REPO_ROOT}/training/tools/convert_to_hf.sh ${RUN_DIR} open-dcoder-ablation-0.1-ctw0.1"
echo "======================================"
