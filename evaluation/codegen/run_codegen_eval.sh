#!/usr/bin/env bash
# Code-generation evaluation of one Open-dCoder checkpoint (paper Table 5 top, Appendix Table 9):
#   HumanEval, HumanEval+, MBPP, MBPP+; Pass@1 and Pass@10 (n = 10 samples, T = 0.8, top-k 200, 128 steps)
#   decoders: vanilla (alg p2_upgraded) and ReMDM (alg p2_upgraded_ReMDM)
# Replays the paper's 4-process lm-eval run as 4 virtual ranks, so the samples of the published
# models are identical to the recorded runs (see README.md).
#
# usage: bash evaluation/codegen/run_codegen_eval.sh MODEL OUT_DIR [GPUS] [PROCS_PER_GPU] [ALGS] [TASKS]
#   MODEL          local HF checkpoint dir, Hub id, or Hub id@revision (downloaded at that revision)
#   GPUS           comma list of GPU ids (default 0); the 4 virtual ranks x decoders x tasks run as
#                  independent single-GPU workers, GPUS x PROCS_PER_GPU at a time (results do not depend on it)
#   PROCS_PER_GPU  default 4 (a 0.5B HumanEval worker uses about 6 GB)
#   ALGS           vanilla | remdm | vanilla,remdm (default)
#   TASKS          subset of humaneval,humaneval_plus,mbpp,mbpp_plus (default: all four)
# The same settings can be given as environment variables (GPUS=... PROCS_PER_GPU=... ALGS=... TASKS=...).
# MAX_DOCS=N runs only the first N docs of every virtual rank (spot check); PYTHON selects the interpreter.
# output: OUT_DIR/summary.json, metrics_<task>__<alg>.json, samples_<task>__<alg>.jsonl, shards/*.log
set -euo pipefail
MODEL="${1:?usage: run_codegen_eval.sh MODEL OUT_DIR [GPUS] [PROCS_PER_GPU] [ALGS] [TASKS]}"
OUT="${2:?usage: run_codegen_eval.sh MODEL OUT_DIR [GPUS] [PROCS_PER_GPU] [ALGS] [TASKS]}"
GPUS="${3:-${GPUS:-0}}"
PPG="${4:-${PROCS_PER_GPU:-4}}"
ALGS="${5:-${ALGS:-vanilla,remdm}}"
TASKS="${6:-${TASKS:-humaneval,humaneval_plus,mbpp,mbpp_plus}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
EXTRA=()
if [ -n "${MAX_DOCS:-}" ]; then EXTRA+=(--max_docs "$MAX_DOCS"); fi

# the repository's Open-dLLM eval code, vendored lm-evaluation-harness and veomni; nothing else
export PYTHONPATH="$REPO/Open-dLLM/eval/eval_completion:$REPO/Open-dLLM/lm-evaluation-harness:$REPO/Open-dLLM"
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
"$PY" "$HERE/codegen_eval.py" run --model "$MODEL" --out_dir "$OUT" --gpus "$GPUS" \
    --procs_per_gpu "$PPG" --algs "$ALGS" --tasks "$TASKS" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$OUT/run.log"
