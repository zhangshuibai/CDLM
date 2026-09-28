#!/bin/bash
# =============================================================================
# CRB evaluation (error localisation + correction) of one Open-dCoder-family model.
#
#   bash evaluation/crb/run_crb.sh <MODEL> <LABEL> [options]
#   bash evaluation/crb/run_crb.sh --help
#
#   MODEL  Hub id whose name contains "open-dcoder" (e.g.
#          Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000) or a local HF-format
#          directory (safetensors weights, config.json, tokenizer files).  Local
#          directories are symlinked to <out_dir>/models/open-dcoder-0.5B-<LABEL>,
#          because utils.py selects the Open-dCoder path (bidirectional attention,
#          logit shift, mask id 151665) by the substring "open-dcoder" in the name.
#   LABEL  output prefix, [A-Za-z0-9_.-]+, must not contain llada or dream.
#
# Options:
#   --gpus 0               comma-separated GPU ids
#   --jobs 3               concurrent refinement processes, one GPU each; job slot i runs on
#                          GPU i mod (number of --gpus), so several jobs can share a GPU
#   --port_base 29500      rendezvous ports port_base .. port_base+jobs-1
#   --eval_jobs 8          concurrent CPU evaluation processes
#   --nr "1 2 3 4 5"       corruption levels (n_replace)
#   --steps "2 3 4 5"      refined_steps S; T = S-1 correction rounds
#   --datasets "human-eval human-eval+ mbpp mbpp+"
#   --error_types "operator var literal"
#   --out_dir DIR          default evaluation/crb/outputs
#   --model_revision REV   Hub revision of MODEL (the snapshot is then used as a local dir)
#   --phase all|refine|eval|metrics|preflight
#                          preflight: only the checks below (code, packages, inputs, model,
#                          tokenizer, evaluation data, code_eval); CPU only, exit 0 if all pass
#   --purge                delete histories and refined jsonl once the summary is complete
#   --allow_data_drift     run even if the evaluation datasets differ from the pinned ones
#
# Environment:
#   CRB_PYTHON           interpreter (default: python on PATH)
#   CRB_INPUTS_DIR       local copy of the input set; otherwise the paper's set is
#                        downloaded from the Hub dataset CRB_INPUTS_REPO
#                        (default Shuibai12138/crb-paper-inputs, directory
#                        open-dcoder-0.5B/) at CRB_INPUTS_REVISION (default: the pinned upload
#                        21cae17423b073b152e997746876d6b828b18358).
#                        Accepted layouts: DIR/open-dcoder-0.5B/<dataset>/evaluated/,
#                        DIR/buggy_datasets/<dataset>/evaluated/ or DIR/<dataset>/evaluated/.
#                        A set written by build_crb_inputs.sh (DIR/crb_inputs_meta.json)
#                        is checked against its own INPUTS.md5 and tokenizer hash.
#   CRB_OFFLINE=1        never contact the Hub (model, inputs and datasets must be cached)
#   HF_SCRIPTS_VERSION   code_eval metric version fetched by evaluate (default v0.4.0)
#
# Preflight: fails on a missing liger-kernel (without it veomni's Qwen2 runs plain PyTorch
# layers and the numerics change), a code_eval module other than the verified one, or
# inputs, model or tokenizer that do not match; warns if torch, transformers, tokenizers,
# liger-kernel, triton, accelerate, datasets, evaluate or huggingface-hub differ from the
# verified versions (evaluation/ENVIRONMENT.md).
#
# Protocol: refine_setting remove_all, algorithm self_conf-remask:vanilla, remask iff
# confidence <= 0.9, temperature 0.0, batch_size 1, refined_steps 2..5 (T = 1..4; the
# last step never remasks).  Evaluation: evaluate_code.py --no_postprocess, 8 s timeout.
#
# Output: <out_dir>/runs/<LABEL>_{results,history}/...   (build_output_paths layout)
#         <out_dir>/results/<LABEL>_summary.json, _per_cell.csv, _per_sample.csv
#         <out_dir>/logs/<LABEL>/
# Resumable: refinement uses --skip_existing, evaluation --skip_if_exist.
# =============================================================================
set -uo pipefail
{  # parsed as a whole, so editing this file cannot affect a running evaluation
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# ------------------------------ protocol --------------------------------------
ALGORITHM="self_conf-remask:vanilla"
ALG_SUFFIX="self_conf-remask_vanilla_ct090_t00"
REFINE_SETTING="remove_all"
CT=0.9
TEMPERATURE=0.0
BATCH_SIZE=1

# ---------------------------- paper input set ---------------------------------
PAPER_TAG="Open-Dcoder-0.5B-mixture-mdm-step2000"
PAPER_DATA_NUM=2
PAPER_MANIFEST=$HERE/paper_inputs.md5
PAPER_MANIFEST_MD5=5a5be8972e7066c7fbe2d7c92dbfc1b8
PAPER_TOKEN_SHA=39197c1ad25f664d2a7da602507e896bba1c83348e40ad51ce979f0aa1f5de0c
INPUTS_REPO=${CRB_INPUTS_REPO:-Shuibai12138/crb-paper-inputs}
INPUTS_REVISION=${CRB_INPUTS_REVISION:-21cae17423b073b152e997746876d6b828b18358}
INPUTS_SUBDIR=open-dcoder-0.5B

# pipeline files verified to reproduce the paper's CRB numbers (a mismatch only warns)
PIPE_SHA=(
  "001954469b88caa2c5ab7231e68eebf7fa36a9e2a1056c42b55971517753f5ba  refine_code.py"
  "2eb54cd0948bee62ed004ad04088139e4e35285eca5f2123a34664a03e3a47b6  llada_sample.py"
  "51a76aa0595a2aa594211d1b225ea3a3d0e3e46a9de95899646ff1e89c30b6d0  utils.py"
  "f34213f15ff1f9590a1642059e7670182e322e18d6a1d24cc096f771ca42f20b  evaluate_code.py"
  "14df0efd6d31861d3d7b3c39aa2ed595f9482436596c3de1ebbfb639ca22abb0  sanitize.py"
)
VEOMNI_TREE_SHA=ae77fb8ca0c0aa33a1cd6cf0abf19d6a14e4358911d6fd5a9ccc430580cd6214

# ------------------------------- arguments ------------------------------------
usage() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac
[ $# -ge 2 ] || { usage; exit 2; }
MODEL=$1; LABEL=$2; shift 2
GPUS=0; JOBS=3; PORT_BASE=29500; EVAL_JOBS=8
NR_LIST="1 2 3 4 5"; STEPS_LIST="2 3 4 5"
DATASETS_LIST="human-eval human-eval+ mbpp mbpp+"; ERROR_TYPES_LIST="operator var literal"
OUT=$HERE/outputs; MODEL_REV=""; PHASE=all; PURGE=0; ALLOW_DRIFT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    --port_base) PORT_BASE=$2; shift 2;;
    --eval_jobs) EVAL_JOBS=$2; shift 2;;
    --nr) NR_LIST=$2; shift 2;;
    --steps) STEPS_LIST=$2; shift 2;;
    --datasets) DATASETS_LIST=$2; shift 2;;
    --error_types) ERROR_TYPES_LIST=$2; shift 2;;
    --out_dir) OUT=$2; shift 2;;
    --model_revision) MODEL_REV=$2; shift 2;;
    --phase) PHASE=$2; shift 2;;
    --purge) PURGE=1; shift;;
    --allow_data_drift) ALLOW_DRIFT=1; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
fail() { echo "PREFLIGHT FAIL: $*"; exit 1; }
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "bad LABEL '$LABEL'"
case "${LABEL,,}" in *llada*|*dream*) fail "LABEL must not contain llada/dream";; esac
case "$PHASE" in all|refine|eval|metrics|preflight) ;; *) fail "bad --phase $PHASE";; esac
[[ "$JOBS" =~ ^[1-9][0-9]*$ && "$EVAL_JOBS" =~ ^[1-9][0-9]*$ && "$PORT_BASE" =~ ^[0-9]+$ ]] || fail "--jobs/--eval_jobs/--port_base must be positive integers"
IFS=, read -r -a GPU_ARR <<< "$GPUS"
DATASETS=($DATASETS_LIST); ERROR_TYPES=($ERROR_TYPES_LIST); NR_ARR=($NR_LIST); STEPS_ARR=($STEPS_LIST)
for d in "${DATASETS[@]}"; do case "$d" in human-eval|human-eval+|mbpp|mbpp+) ;; *) fail "bad dataset $d";; esac; done
for e in "${ERROR_TYPES[@]}"; do case "$e" in operator|var|literal) ;; *) fail "bad error type $e";; esac; done
for s in "${STEPS_ARR[@]}"; do [[ "$s" =~ ^[0-9]+$ ]] && [ "$s" -ge 2 ] || fail "refined_steps must be >= 2 (got $s)"; done
# Hub ids are passed to utils.py unchanged: it routes on "open-dcoder" in model_name.lower()
# and switches to the LLaDA mask id if the name contains "llada".
LC_ARG=${MODEL,,}
if [ ! -d "$MODEL" ]; then
  case "${LC_ARG%/}" in
    shuibai12138/cdlm-0.5b)
      fail "Shuibai12138/CDLM-0.5B has the same weights as Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000, but its name lacks 'open-dcoder', so utils.py would not load it as an Open-dCoder model. Run: bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 $LABEL";;
  esac
  [[ "$LC_ARG" == *open-dcoder* ]] || fail "Hub id '$MODEL' lacks 'open-dcoder', so utils.py would not load it as an Open-dCoder model; for an Open-dCoder-family checkpoint published under another name, pass a local download of it"
  [[ "$LC_ARG" == *llada* || "$LC_ARG" == *dream-v0* ]] && fail "model name contains llada/dream-v0"
fi

PY=${CRB_PYTHON:-python}
command -v "$PY" >/dev/null || fail "python interpreter '$PY' not found (set CRB_PYTHON)"
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
# evaluate 0.4.5 would ask for tag v0.4.5 of the code_eval Space, which does not exist, and fall
# back to its moving main; v0.4.0 is the verified module (the md5 check below is still fatal)
export HF_SCRIPTS_VERSION=${HF_SCRIPTS_VERSION:-v0.4.0}
export PYTHONPATH=$REPO/Open-dLLM
go_offline() { export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1; }
[ "${CRB_OFFLINE:-0}" = 1 ] && go_offline

mkdir -p "$OUT" || fail "cannot create $OUT"
OUT=$(cd "$OUT" && pwd)
WORK=$OUT/runs; LOGDIR=$OUT/logs/$LABEL; RESDIR=$OUT/results
mkdir -p "$WORK" "$LOGDIR" "$RESDIR" "$OUT/models"
META=$LOGDIR/run_meta.json
ts() { date +%s; }
T_START=$(ts)

# ------------------------------- preflight ------------------------------------
cd "$REPO" || fail "no repo dir"
for f in refine_code.py llada_sample.py utils.py evaluate_code.py sanitize.py; do
  [ -f "$f" ] || fail "$REPO/$f missing"
done
PIPE_OK=1
for l in "${PIPE_SHA[@]}"; do echo "$l"; done | sha256sum -c --quiet - >/dev/null 2>&1 || PIPE_OK=0
[ $PIPE_OK = 1 ] || echo "WARNING: pipeline files differ from the release verified against the paper's numbers"
[ -d "$REPO/Open-dLLM/veomni" ] || fail "$REPO/Open-dLLM/veomni missing"
VEOMNI_SHA=$(cd "$REPO/Open-dLLM" && find veomni -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-64)
[ "$VEOMNI_SHA" = "$VEOMNI_TREE_SHA" ] || echo "WARNING: Open-dLLM/veomni differs from the verified tree"
"$PY" -c "import torch, transformers, datasets, evaluate, huggingface_hub, numpy" \
  || fail "missing python packages (torch transformers datasets evaluate huggingface_hub numpy)"
# versions of the packages that set the numerics; liger-kernel is detected the way veomni does it
PKG_CHECK=$("$PY" - <<'PY'
import importlib.metadata as md, importlib.util, json
want = {"torch": "2.5.0+cu121", "transformers": "4.54.1", "tokenizers": "0.21.4", "liger-kernel": "0.5.8",
        "triton": "3.1.0", "accelerate": "1.10.1", "datasets": "3.6.0", "evaluate": "0.4.5",
        "huggingface-hub": "0.34.4"}
got = {}
for p, w in want.items():
    try:
        got[p] = md.version(p)
    except md.PackageNotFoundError:
        got[p] = None
    if got[p] != w:
        print("DIFF", p, got[p], w)
if importlib.util.find_spec("liger_kernel") is None:
    print("MISSING liger-kernel")
print(json.dumps(got))
PY
) || fail "could not read package versions"
PKG_VERSIONS=$(echo "$PKG_CHECK" | tail -1)
grep -q '^MISSING liger-kernel' <<< "$PKG_CHECK" \
  && fail "liger-kernel is not importable: veomni's Qwen2 would silently run plain PyTorch layers and change the numerics (pip install liger-kernel==0.5.8; see evaluation/ENVIRONMENT.md)"
while read -r _ p g w; do
  echo "WARNING: $p $g differs from the verified $w (evaluation/ENVIRONMENT.md); results may not reproduce bitwise"
done < <(grep '^DIFF' <<< "$PKG_CHECK")
VEOMNI_FILE=$(cd "$WORK" && "$PY" -c "import veomni;print(veomni.__file__)" 2>/dev/null | tail -1)
[ "$VEOMNI_FILE" = "$REPO/Open-dLLM/veomni/__init__.py" ] || fail "veomni imported from '$VEOMNI_FILE', expected $REPO/Open-dLLM/veomni"

# inputs
if [ -n "${CRB_INPUTS_DIR:-}" ]; then
  [ -d "$CRB_INPUTS_DIR" ] || fail "CRB_INPUTS_DIR=$CRB_INPUTS_DIR is not a directory"
  IN_BASE=$(cd "$CRB_INPUTS_DIR" && pwd)
  INPUTS_SOURCE="local:$IN_BASE"
else
  echo "[preflight] fetching $INPUTS_REPO@$INPUTS_REVISION ($INPUTS_SUBDIR/) ..."
  IN_BASE=$("$PY" - "$INPUTS_REPO" "$INPUTS_REVISION" "$INPUTS_SUBDIR" <<'PY' 2>"$LOGDIR/inputs_download.log" | tail -1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], repo_type="dataset", revision=sys.argv[2],
                        allow_patterns=[sys.argv[3] + "/**"]))
PY
)
  [ -d "$IN_BASE" ] || fail "could not download $INPUTS_SUBDIR/ from $INPUTS_REPO@$INPUTS_REVISION (see $LOGDIR/inputs_download.log); or set CRB_INPUTS_DIR"
  INPUTS_SOURCE="hub:$INPUTS_REPO@$INPUTS_REVISION:$(basename "$IN_BASE")"
fi
IN_ROOT=""
for c in "$IN_BASE/$INPUTS_SUBDIR" "$IN_BASE/buggy_datasets" "$IN_BASE"; do
  [ -d "$c/${DATASETS[0]}/evaluated" ] && { IN_ROOT=$(cd "$c" && pwd -P); break; }
done
[ -n "$IN_ROOT" ] || fail "no <dataset>/evaluated/ directories under $IN_BASE"
if [ -L "$WORK/buggy_datasets" ]; then
  [ "$(readlink -f "$WORK/buggy_datasets")" = "$IN_ROOT" ] \
    || fail "$WORK/buggy_datasets points to $(readlink -f "$WORK/buggy_datasets"), not $IN_ROOT; use another --out_dir for another input set"
elif [ -e "$WORK/buggy_datasets" ]; then
  fail "$WORK/buggy_datasets exists and is not a symlink"
else
  ln -s "$IN_ROOT" "$WORK/buggy_datasets"
fi
if [ -f "$IN_BASE/crb_inputs_meta.json" ]; then
  # input set written by build_crb_inputs.sh
  PAPER_SET=0
  read -r TAG DATA_NUM WANT_TOKEN_SHA < <("$PY" -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['tag'],m['data_num'],m['token_sha256'])" "$IN_BASE/crb_inputs_meta.json")
  MANIFEST=$IN_BASE/INPUTS.md5
  [ -f "$MANIFEST" ] || fail "$MANIFEST missing"
  MANIFEST_MD5=$(md5sum < "$MANIFEST" | cut -c1-32)
  TOK_ARGS=(--tag "$TAG" --data_num "$DATA_NUM")
  read -r -a M_ARGS < <("$PY" -c "import json,sys;m=json.load(open(sys.argv[1]));print('--nr',*m['nr'],'--datasets',*m['datasets'],'--error_types',*m['error_types'])" "$IN_BASE/crb_inputs_meta.json")
  TOK_ARGS+=("${M_ARGS[@]}")
else
  PAPER_SET=1
  TAG=$PAPER_TAG; DATA_NUM=$PAPER_DATA_NUM; WANT_TOKEN_SHA=$PAPER_TOKEN_SHA
  MANIFEST=$PAPER_MANIFEST
  MANIFEST_MD5=$(md5sum < "$MANIFEST" | cut -c1-32)
  [ "$MANIFEST_MD5" = "$PAPER_MANIFEST_MD5" ] || fail "paper_inputs.md5 was modified"
  TOK_ARGS=()
fi
(cd "$WORK" && md5sum -c --quiet "$MANIFEST") || fail "input file md5 mismatch against $MANIFEST"
for NR in "${NR_ARR[@]}"; do for DS in "${DATASETS[@]}"; do for ET in "${ERROR_TYPES[@]}"; do
  [ -s "$WORK/buggy_datasets/$DS/evaluated/${TAG}_${ET}_${DATA_NUM}_wrong_${NR}_evaluated.jsonl" ] \
    || fail "input missing: $DS/evaluated/${TAG}_${ET}_${DATA_NUM}_wrong_${NR}_evaluated.jsonl"
done; done; done
echo "[preflight] inputs OK  $INPUTS_SOURCE  ($(grep -c . "$MANIFEST") files, manifest md5 $MANIFEST_MD5)"

# model (local directories are symlinked to an open-dcoder name, see the header)
if [ ! -d "$MODEL" ]; then
  echo "[preflight] resolving $MODEL${MODEL_REV:+@$MODEL_REV} ..."
  SNAPDIR=$("$PY" - "$MODEL" "$MODEL_REV" <<'PY' 2>"$LOGDIR/model_download.log" | tail -1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2] or None))
PY
)
  [ -d "$SNAPDIR" ] || fail "could not fetch $MODEL (see $LOGDIR/model_download.log)"
  if [ -n "$MODEL_REV" ]; then MODEL_DIR=$SNAPDIR; else MODEL_DIR=""; MODEL_NAME=$MODEL; fi
  MODEL_SRC="hub:$MODEL@$(basename "$SNAPDIR")"
  WDIR=$SNAPDIR; CFG=$SNAPDIR/config.json
else
  MODEL_DIR=$MODEL; MODEL_SRC="local:$(readlink -f "$MODEL")"
fi
if [ -n "$MODEL_DIR" ]; then
  SRC=$(readlink -f "$MODEL_DIR")
  [ -s "$SRC/model.safetensors" ] || [ -s "$SRC/model.safetensors.index.json" ] || fail "no model.safetensors in $SRC"
  [ -s "$SRC/config.json" ] || fail "$SRC/config.json missing"
  ls "$SRC"/tokenizer_config.json "$SRC"/vocab.json "$SRC"/merges.txt >/dev/null 2>&1 || fail "tokenizer files missing in $SRC"
  MODEL_NAME=$OUT/models/open-dcoder-0.5B-$LABEL
  ln -sfn "$SRC" "$MODEL_NAME"
  WDIR=$SRC; CFG=$SRC/config.json
fi
LC=${MODEL_NAME,,}
[[ "$LC" == *open-dcoder* ]] || fail "model name '$MODEL_NAME' lacks 'open-dcoder' (utils.py routing)"
[[ "$LC" == *llada* || "$LC" == *dream-v0* ]] && fail "model name '$MODEL_NAME' contains llada/dream-v0 (e.g. in --out_dir)"
"$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));assert c.get('model_type')=='qwen2',c.get('model_type')" "$CFG" \
  || fail "config.json model_type is not qwen2 ($CFG)"
echo "[preflight] computing weight sha256 ..."
if [ -s "$WDIR/model.safetensors" ]; then
  WEIGHTS_SHA=$(sha256sum "$(readlink -f "$WDIR/model.safetensors")" | cut -c1-64)
else  # sharded checkpoint: sha256 over the per-shard hashes
  WEIGHTS_SHA=$(cd "$WDIR" && for f in $(ls *.safetensors | sort); do
    echo "$(sha256sum "$(readlink -f "$f")" | cut -c1-64)  $f"; done | sha256sum | cut -c1-64)
fi
TOK=$(cd "$WORK" && "$PY" "$HERE/crb_tokcheck.py" "$MODEL_NAME" "$WORK" "${TOK_ARGS[@]}" 2>"$LOGDIR/tokcheck.log" | tail -1)
echo "$TOK" | grep -q "\"token_sha256\": \"$WANT_TOKEN_SHA\"" || fail "tokenizer produces different token ids on the inputs: $TOK"
echo "$TOK" | grep -q '"n_error_pos_out_of_range": 0' || fail "error positions out of range: $TOK"
echo "$TOK" | grep -q '"pad_id": 151643' || fail "pad id differs: $TOK"
echo "$TOK" | grep -q '"id151665": "<M>"' || fail "mask token 151665 is not <M>: $TOK"

# evaluation datasets: cache the pinned revisions, then everything runs offline
if [ "${CRB_OFFLINE:-0}" != 1 ]; then
  "$PY" "$HERE/crb_evaldata.py" prefetch --datasets "${DATASETS[@]}" > "$LOGDIR/evaldata_prefetch.log" 2>&1 \
    || fail "could not fetch the evaluation datasets (see $LOGDIR/evaldata_prefetch.log)"
fi
go_offline
EVALDATA=$(cd "$WORK" && "$PY" "$HERE/crb_evaldata.py" fingerprint --datasets "${DATASETS[@]}" 2>"$LOGDIR/evaldata.log" | tail -1)
if ! echo "$EVALDATA" | grep -q '"mismatch": \[\]'; then
  [ $ALLOW_DRIFT = 1 ] && echo "WARNING: evaluation datasets differ from the pinned revisions: $EVALDATA" \
    || fail "evaluation datasets differ from the pinned revisions (see crb_evaldata.py; --allow_data_drift to continue): $EVALDATA"
fi
echo "$EVALDATA" | grep -q '"code_eval_matches": true' \
  || fail "the code_eval metric module differs from the verified one (HF_SCRIPTS_VERSION=$HF_SCRIPTS_VERSION; see $LOGDIR/evaldata.log): $EVALDATA"
if [ "$PHASE" = preflight ]; then
  echo "[preflight] OK  model_name=$MODEL_NAME  weights_sha256=$WEIGHTS_SHA  (--phase preflight: no GPU work, run metadata not written)"
  exit 0
fi

CRBMETA_model_arg=$MODEL CRBMETA_model_revision=$MODEL_REV CRBMETA_model_source=$MODEL_SRC CRBMETA_model_name_used=$MODEL_NAME \
CRBMETA_weights_sha256=$WEIGHTS_SHA CRBMETA_label=$LABEL CRBMETA_gpus=$GPUS CRBMETA_jobs=$JOBS CRBMETA_eval_jobs=$EVAL_JOBS \
CRBMETA_nr=$NR_LIST CRBMETA_steps=$STEPS_LIST CRBMETA_datasets=$DATASETS_LIST CRBMETA_error_types=$ERROR_TYPES_LIST \
CRBMETA_algorithm=$ALGORITHM CRBMETA_tau=$CT CRBMETA_temperature=$TEMPERATURE CRBMETA_batch_size=$BATCH_SIZE \
CRBMETA_refine_setting=$REFINE_SETTING CRBMETA_inputs_source=$INPUTS_SOURCE CRBMETA_inputs_tag=$TAG CRBMETA_inputs_data_num=$DATA_NUM \
CRBMETA_paper_input_set=$PAPER_SET CRBMETA_inputs_manifest_md5=$MANIFEST_MD5 CRBMETA_token_sha256=$WANT_TOKEN_SHA \
CRBMETA_pipeline_matches_verified_release=$PIPE_OK CRBMETA_veomni_tree_sha256=$VEOMNI_SHA CRBMETA_evaldata=$EVALDATA \
CRBMETA_package_versions=$PKG_VERSIONS CRBMETA_hf_scripts_version=$HF_SCRIPTS_VERSION \
"$PY" - "$META" "$REPO" "${GPU_ARR[0]}" <<'PY' || fail "could not write run metadata"
import json, os, platform, subprocess, sys
import torch, transformers


def sh(*c):
    try:
        return subprocess.run(list(c), capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


meta, repo, gpu = sys.argv[1:4]
m = {k[8:]: v for k, v in os.environ.items() if k.startswith("CRBMETA_")}
for k in ["jobs", "eval_jobs", "batch_size", "inputs_data_num"]:
    m[k] = int(m[k])
for k in ["tau", "temperature"]:
    m[k] = float(m[k])
for k in ["paper_input_set", "pipeline_matches_verified_release"]:
    m[k] = m[k] == "1"
m["evaldata"] = json.loads(m["evaldata"])
m["package_versions"] = json.loads(m["package_versions"])
m["repo_commit"] = sh("git", "-C", repo, "rev-parse", "HEAD")
m["repo_pipeline_modified"] = bool(sh("git", "-C", repo, "status", "--porcelain", "--", "refine_code.py",
                                      "llada_sample.py", "utils.py", "evaluate_code.py", "sanitize.py",
                                      "Open-dLLM/veomni"))
m.update({"python": platform.python_version(), "torch": torch.__version__,
          "transformers": transformers.__version__,
          "gpu_name": sh("nvidia-smi", "-i", gpu, "--query-gpu=name", "--format=csv,noheader"),
          "host": platform.node(), "started": sh("date", "-Is")})
json.dump(dict(sorted(m.items())), open(meta, "w"), indent=1)
PY
echo "[preflight] OK  model_name=$MODEL_NAME  weights_sha256=$WEIGHTS_SHA"

stem() { echo "${TAG}_${1}_${DATA_NUM}_wrong_${2}_evaluated"; }
rdir() { echo "${LABEL}_results/refined_steps${1}/${REFINE_SETTING}/${ALG_SUFFIX}/buggy_datasets/${2}/evaluated/$(stem $3 $4)"; }
cd "$WORK" || exit 1

# ------------------------------ phase 1: refine --------------------------------
if [ "$PHASE" = all ] || [ "$PHASE" = refine ]; then
  T0=$(ts); NFAIL=0
  declare -a SLOT_PID; for i in $(seq 0 $((JOBS-1))); do SLOT_PID[$i]=0; done
  get_slot() {  # blocks until a slot is free; echoes slot index
    while true; do
      for i in $(seq 0 $((JOBS-1))); do
        p=${SLOT_PID[$i]}
        if [ "$p" = 0 ] || ! kill -0 "$p" 2>/dev/null; then echo $i; return; fi
      done
      sleep 1
    done
  }
  for NR in "${NR_ARR[@]}"; do for DS in "${DATASETS[@]}"; do for ET in "${ERROR_TYPES[@]}"; do
    ST=$(stem $ET $NR); IN="buggy_datasets/${DS}/evaluated/${ST}.jsonl"
    for S in "${STEPS_ARR[@]}"; do
      RD=$(rdir $S $DS $ET $NR)
      [ -s "$RD/${ST}_results_refined.jsonl" ] && continue
      SL=$(get_slot); PORT=$((PORT_BASE+SL)); G=${GPU_ARR[$((SL % ${#GPU_ARR[@]}))]}
      LOG=$LOGDIR/refine_${DS}_${ET}_nr${NR}_s${S}.log
      CUDA_VISIBLE_DEVICES=$G "$PY" -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
          "$REPO/refine_code.py" \
          --initial_results_file "$IN" --model_name "$MODEL_NAME" --batch_size $BATCH_SIZE \
          --refined_steps $S --algorithm "$ALGORITHM" --temperature $TEMPERATURE \
          --refine_setting $REFINE_SETTING --confidence_threshold $CT \
          --output_prefix "$LABEL" --skip_existing > "$LOG" 2>&1 &
      SLOT_PID[$SL]=$!
      echo "[refine] slot $SL gpu $G port $PORT  $DS $ET nr=$NR steps=$S (T=$((S-1)))"
    done
  done; done; done
  wait
  for NR in "${NR_ARR[@]}"; do for DS in "${DATASETS[@]}"; do for ET in "${ERROR_TYPES[@]}"; do for S in "${STEPS_ARR[@]}"; do
    RD=$(rdir $S $DS $ET $NR); ST=$(stem $ET $NR)
    [ -s "$RD/${ST}_results_refined.jsonl" ] || { echo "REFINE MISSING: $DS $ET nr=$NR steps=$S (log $LOGDIR/refine_${DS}_${ET}_nr${NR}_s${S}.log)"; NFAIL=$((NFAIL+1)); }
  done; done; done; done
  echo "[refine] done in $(( $(ts)-T0 )) s, missing=$NFAIL"
  echo "{\"refine_seconds\": $(( $(ts)-T0 )), \"refine_missing\": $NFAIL}" > "$LOGDIR/refine_timing.json"
fi

# ------------------------- phase 2: execution eval (CPU) -------------------------
if [ "$PHASE" = all ] || [ "$PHASE" = eval ]; then
  T0=$(ts); N=0
  for NR in "${NR_ARR[@]}"; do for DS in "${DATASETS[@]}"; do for ET in "${ERROR_TYPES[@]}"; do for S in "${STEPS_ARR[@]}"; do
    RD=$(rdir $S $DS $ET $NR); ST=$(stem $ET $NR); RF="$RD/${ST}_results_refined.jsonl"
    [ -s "$RF" ] || continue
    [ -s "$RD/${ST}_results_refined_evaluated.jsonl" ] && [ -s "$RD/pass_at_1_summary.json" ] && continue
    rm -f "$RD/${ST}_results_refined_evaluated.jsonl" "$RD/pass_at_1_summary.json"   # never keep a partial pair
    ( CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/evaluate_code.py" --results_file "$RF" \
        --output_file "$RD/${ST}_results_refined_evaluated.jsonl" --dataset "$DS" --no_postprocess \
        --summary_file "$RD/pass_at_1_summary.json" \
        --summary_metadata "dataset:${DS},error_type:${ET},n_replace:${NR},model_name:${MODEL_NAME},data_num:${DATA_NUM},refined_steps:${S},algorithm:${ALGORITHM},confidence_threshold:${CT},temperature:${TEMPERATURE},refine_setting:${REFINE_SETTING}" \
        --skip_if_exist > "$RD/eval.log" 2>&1 || echo "EVAL FAILED $RD" ) &
    N=$((N+1)); [ $((N % 20)) -eq 0 ] && echo "[eval] launched $N"
    while [ "$(jobs -rp | wc -l)" -ge "$EVAL_JOBS" ]; do wait -n; done
  done; done; done; done
  wait
  echo "[eval] done in $(( $(ts)-T0 )) s ($N jobs)"
  echo "{\"eval_seconds\": $(( $(ts)-T0 )), \"eval_jobs_run\": $N}" > "$LOGDIR/eval_timing.json"
fi

# ------------------------------ phase 3: metrics ---------------------------------
if [ "$PHASE" = refine ]; then
  echo "[done] total $(( $(ts)-T_START )) s"
  [ "$NFAIL" = 0 ] || exit 1
  exit 0
fi
"$PY" - "$META" "$LOGDIR" <<'PY'
import json, sys, glob, os
m = json.load(open(sys.argv[1]))
for f in glob.glob(os.path.join(sys.argv[2], "*_timing.json")):
    m.update(json.load(open(f)))
json.dump(m, open(sys.argv[1], "w"), indent=1)
PY
"$PY" "$HERE/crb_metrics.py" --root "$WORK" --prefix "$LABEL" --nr "${NR_ARR[@]}" --steps "${STEPS_ARR[@]}" \
    --datasets "${DATASETS[@]}" --error_types "${ERROR_TYPES[@]}" --tag "$TAG" --data_num "$DATA_NUM" \
    --outdir "$RESDIR" --meta "$META" || exit 1
echo "[metrics] $RESDIR/${LABEL}_summary.json"
COMPLETE=0
"$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d['complete'] else 1)" "$RESDIR/${LABEL}_summary.json" && COMPLETE=1

if [ "$PURGE" = 1 ]; then
  if [ $COMPLETE = 1 ]; then
    rm -rf "$WORK/${LABEL}_history"
    find "$WORK/${LABEL}_results" -name "*_results_refined.jsonl" -delete
    echo "[purge] deleted histories and refined jsonl (kept *_evaluated.jsonl and results/)"
  else
    echo "[purge] summary incomplete, not purging"
  fi
fi
echo "[done] total $(( $(ts)-T_START )) s"
[ $COMPLETE = 1 ] || { echo "summary incomplete (see missing/problems in $RESDIR/${LABEL}_summary.json)"; exit 1; }
exit 0
}
