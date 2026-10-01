#!/bin/bash
# Code-generation guard for LLaDA-8B-Base and LoRA adapters: HumanEval+ and MBPP+ pass@1
# (see codegen_guard.py for the protocol).
#   usage: bash run_codegen_guard.sh <adapter|NONE> <out_dir> [gpus (0)] [tasks (humaneval_plus,mbpp_plus)]
#
# <adapter> is NONE (the base model), a local adapter directory or a Hub id
# <owner>/<name>[/<subfolder>][@<revision>]; a Hub adapter is downloaded first. Each task is split into one
# shard per GPU (shard k gets problems k, k+G, k+2G, ...); every GPU runs its shards one after the other, then
# all shards are merged and scored on the CPU (log in <out_dir>/log_merge.txt). Greedy decoding with batch
# size 1 makes the programs independent of the number of GPUs.
#
# Finished shards are reused when the run is repeated with the same <out_dir>, adapter weights, LIMIT and
# GPU count. The out_dir records the adapter by a hash of its files (adapter.txt); shards of other weights,
# another LIMIT or another GPU count stop the run.
#
# Environment overrides: PYTHON [python], LIMIT [unset: all problems; N: the first N of every shard].
# Generated code is executed: run this in an isolated environment.
set -uo pipefail

ADAPTER=${1:?usage: run_codegen_guard.sh <adapter|NONE> <out_dir> [gpus] [tasks]}
OUT=${2:?usage: run_codegen_guard.sh <adapter|NONE> <out_dir> [gpus] [tasks]}
IFS=',' read -r -a GPUS <<< "${3:-0}"
IFS=',' read -r -a TASKS <<< "${4:-humaneval_plus,mbpp_plus}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
G=${#GPUS[@]}

# Resolve the adapter once (a relative path refers to the caller's directory) and identify it by its files.
SPEC="${ADAPTER}"
if [ "${ADAPTER}" = "NONE" ]; then
    ADAPTER_DIR=""; ADAPTER_ID="NONE"
else
    ADAPTER_DIR="$("${PY}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from adapter_path import resolve_adapter; print(resolve_adapter(sys.argv[2]))' \
                   "${SCRIPT_DIR}" "${ADAPTER}" | tail -n 1)" || { echo "!!! cannot resolve adapter ${ADAPTER}"; exit 1; }
    ADAPTER_ID="$("${PY}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from codegen_guard import adapter_digest; print(adapter_digest(sys.argv[2]))' \
                  "${SCRIPT_DIR}" "${ADAPTER_DIR}")" || { echo "!!! cannot hash adapter ${ADAPTER_DIR}"; exit 1; }
fi
ID="${ADAPTER_ID} limit=${LIMIT:-all} shards=${G}"

mkdir -p "${OUT}"
STAMP="${OUT}/adapter.txt"
shopt -s nullglob
EXISTING=("${OUT}"/*/shard*of*.jsonl)
shopt -u nullglob
if [ "${#EXISTING[@]}" -gt 0 ] && [ "$(head -n 1 "${STAMP}" 2>/dev/null)" != "${ID}" ]; then
    echo "!!! ${OUT} holds shards of '$(head -n 1 "${STAMP}" 2>/dev/null)', not '${ID}'; use another out_dir"
    exit 1
fi
printf '%s\n%s\n' "${ID}" "${SPEC} ${ADAPTER_DIR}" > "${STAMP}"

ADAPTER_ARG=()
if [ -n "${ADAPTER_DIR}" ]; then ADAPTER_ARG=(--adapter "${ADAPTER_DIR}" --adapter_spec "${SPEC}"); fi
LIMIT_ARG=()
if [ -n "${LIMIT:-}" ]; then LIMIT_ARG=(--limit "${LIMIT}"); fi

PIDS=()
for k in "${!GPUS[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES=${GPUS[$k]}
        for T in "${TASKS[@]}"; do
            if [ -f "${OUT}/${T}/shard${k}of${G}.jsonl" ]; then echo "reuse ${T} shard ${k}/${G}"; continue; fi
            "${PY}" "${SCRIPT_DIR}/codegen_guard.py" run --task "${T}" --out "${OUT}" --shard "${k}" --num_shards "${G}" \
                ${ADAPTER_ARG[@]+"${ADAPTER_ARG[@]}"} ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} \
                > "${OUT}/log_${T}_shard${k}.txt" 2>&1 || { echo "!!! FAILED ${T} shard ${k} (see ${OUT}/log_${T}_shard${k}.txt)"; exit 1; }
            echo "done ${T} shard ${k}/${G}"
        done
    ) &
    PIDS+=($!)
done
STATUS=0
for p in "${PIDS[@]}"; do wait "${p}" || STATUS=1; done
[ "${STATUS}" -eq 0 ] || { echo "=== SOME SHARDS FAILED ==="; exit 1; }
if "${PY}" "${SCRIPT_DIR}/codegen_guard.py" merge --out "${OUT}" --num_shards "${G}" --tasks "${TASKS[@]}" \
        > "${OUT}/log_merge.txt" 2>&1; then
    grep "^\[guard\]" "${OUT}/log_merge.txt"
else
    echo "!!! MERGE FAILED (${OUT}/log_merge.txt):"; tail -n 20 "${OUT}/log_merge.txt"; exit 1
fi
