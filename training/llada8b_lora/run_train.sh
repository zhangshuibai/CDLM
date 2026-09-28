#!/bin/bash
# G6: matched MDLM vs CDLM LoRA fine-tuning of LLaDA-8B-Base.
#   usage: bash run_train.sh <arm: mdlm|cdlm> <max_steps> <tag>
#
# Environment overrides (defaults in brackets):
#   CUDA_VISIBLE_DEVICES [0,1,2,3]    NPROC [4]    MASTER_PORT [29642]
#   DATA_DIR   [<repo>/data/Nemotron-SFT-Code]
#   OUTPUT_DIR [<repo>/outputs/llada8b_lora]   -> run dir is ${OUTPUT_DIR}/${TAG}_${ARM}
#   TORCHRUN   [torchrun]
set -euo pipefail

ARM=$1
STEPS=$2
TAG=$3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TORCHRUN=${TORCHRUN:-torchrun}
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29642}
DATA=${DATA_DIR:-${REPO_ROOT}/data/Nemotron-SFT-Code}
OUT=${OUTPUT_DIR:-${REPO_ROOT}/outputs/llada8b_lora}/${TAG}_${ARM}

mkdir -p "${OUT}"

# ---- config shared byte-identically by both arms ---------------------------
#  seed 42 | lr 3e-4 | wd 0.01 | cosine over a 20000-step horizon with 1%
#  (=200 step) warmup, so LR at step 2000 is ~98.7% of peak (2.96e-4).
#  NOTE: this is *not* the published 0.5B schedule -- there, veomni derived
#  _train_steps=20,345,053 from train_size=1e12, so lr_warmup_steps=20,345 and
#  step 2000 was still in linear warmup at 9.83% of peak (2.949e-05). We use the
#  more conventional schedule; it is identical across both arms either way.
#  LoRA r=64 a=128 drop=0.05 on
#  q,k,v,attn_out(o),ff_proj(gate),up,ff_out(down) inside transformer blocks only
#  (lm_head + embeddings frozen, mirroring --freeze_layers lm_head,embed_tokens).
"${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_llada_mixture.py" \
    --arm "${ARM}" \
    --data_dir "${DATA}" \
    --output_dir "${OUT}" \
    --max_steps "${STEPS}" \
    --cosine_horizon 20000 \
    --warmup_ratio 0.01 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --micro_batch_size 1 \
    --global_batch_size 12 \
    --max_seq_len 4096 \
    --seed 42 \
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --clean_token_wt 0.0 \
    --save_steps 0 \
    --log_every 20 2>&1 | tee "${OUT}/train.log"
