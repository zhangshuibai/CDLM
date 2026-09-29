#!/bin/bash
# LLaDA-8B-Base LoRA on the public nvidia/OpenCodeInstruct corpus instead of Nemotron-SFT-Code.
#   usage: bash run_train_opencodeinstruct.sh <arm: mdlm|cdlm> [max_steps (2000)] [tag (oci)]
#
# The trainer, its flags and the seed are those of run_train.sh, which this script calls; only
# the training data differ. Render the corpus first with the script shared with the 0.5B runs:
#
#   python training/data_prep/prepare_opencodeinstruct.py
#
# It writes each rendered shard to its own directory and lists those directories in
# data/OpenCodeInstruct-text/train_path.txt. train_llada_mixture.py reads every *.parquet file of
# one directory, so this script links the listed shards into one directory (OCI_FLAT_DIR) under
# their original file names and passes that directory to run_train.sh as DATA_DIR. The trainer
# sorts the file names and shuffles them with the seed, so the data order does not depend on the
# order of train_path.txt or on the filesystem.
#
# Environment overrides (defaults in brackets), in addition to those of run_train.sh:
#   OCI_TRAIN_PATH_FILE [<repo>/data/OpenCodeInstruct-text/train_path.txt]
#   OCI_FLAT_DIR        [<directory of OCI_TRAIN_PATH_FILE>-flat, e.g. data/OpenCodeInstruct-text-flat]
#   OCI_ALLOW_SUBSET    [0]  set to 1 to train on a --shards subset (smoke tests only)
#   PYTHON              [python]  used to read the shard metadata
set -euo pipefail

ARM=${1:?usage: run_train_opencodeinstruct.sh <mdlm|cdlm> [max_steps] [tag]}
STEPS=${2:-2000}
TAG=${3:-oci}
case "${ARM}" in mdlm|cdlm) ;; *) echo "arm must be mdlm or cdlm (got ${ARM})"; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-python}
N_SHARDS=50

LIST=${OCI_TRAIN_PATH_FILE:-${REPO_ROOT}/data/OpenCodeInstruct-text/train_path.txt}
if [ ! -f "${LIST}" ]; then
    echo "missing ${LIST}; run: python training/data_prep/prepare_opencodeinstruct.py"
    exit 1
fi
LIST="$(realpath -- "${LIST}")"
FLAT=${OCI_FLAT_DIR:-$(dirname "${LIST}")-flat}

IFS=',' read -r -a DIRS <<< "$(tr -d '\n' < "${LIST}")"
if [ "${#DIRS[@]}" -ne "${N_SHARDS}" ]; then
    if [ "${OCI_ALLOW_SUBSET:-0}" = 1 ]; then
        echo "WARNING: ${LIST} lists ${#DIRS[@]} shard directories, not ${N_SHARDS} (OCI_ALLOW_SUBSET=1)"
    else
        echo "${LIST} lists ${#DIRS[@]} shard directories, expected ${N_SHARDS}" \
             "(a --shards subset? set OCI_ALLOW_SUBSET=1 for a smoke test)"
        exit 1
    fi
fi

mkdir -p "${FLAT}"
WANT=()
for d in "${DIRS[@]}"; do
    shopt -s nullglob
    files=("${d}"/*.parquet)
    shopt -u nullglob
    if [ "${#files[@]}" -ne 1 ]; then
        echo "${d} must hold exactly one .parquet file (found ${#files[@]})"
        exit 1
    fi
    src="$(realpath -- "${files[0]}")"
    name="$(basename -- "${src}")"
    WANT+=("${name}")
    link="${FLAT}/${name}"
    if [ -L "${link}" ]; then
        [ "$(readlink -- "${link}")" = "${src}" ] || ln -sfn -- "${src}" "${link}"
    elif [ -e "${link}" ]; then
        echo "${link} exists and is not a symlink; remove it or set OCI_FLAT_DIR"
        exit 1
    else
        ln -s -- "${src}" "${link}"
    fi
done
if [ "$(printf '%s\n' "${WANT[@]}" | sort | uniq -d)" != "" ]; then
    echo "${LIST} lists two shards with the same file name"
    exit 1
fi
# The trainer reads every *.parquet file in the directory, so nothing else may be there.
EXTRA="$(comm -23 <(cd "${FLAT}" && ls -1 -- *.parquet 2>/dev/null | sort) \
                  <(printf '%s\n' "${WANT[@]}" | sort))"
if [ -n "${EXTRA}" ]; then
    echo "${FLAT} holds .parquet files that ${LIST} does not list:" ${EXTRA}
    exit 1
fi

# Record which rendered shards the run reads (file name, rows, source sha256, template).
OUT=${OUTPUT_DIR:-${REPO_ROOT}/outputs/llada8b_lora}/${TAG}_${ARM}
mkdir -p "${OUT}"
"${PY}" - "${FLAT}" "${OUT}/data_shards.tsv" <<'EOF'
import json, os, sys
import pyarrow.parquet as pq
flat, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    f.write("file\trows\tsource\trevision\tsource_sha256\ttemplate\n")
    for name in sorted(n for n in os.listdir(flat) if n.endswith(".parquet")):
        pf = pq.ParquetFile(os.path.join(flat, name))
        info = json.loads(pf.schema_arrow.metadata[b"cdlm_render"])
        f.write("\t".join([name, str(pf.metadata.num_rows), info["source"], info["revision"],
                           info["source_sha256"], json.dumps(info["template"])]) + "\n")
EOF
echo "OpenCodeInstruct: ${#WANT[@]} rendered shards linked in ${FLAT} (listed in ${OUT}/data_shards.tsv)"

DATA_DIR="${FLAT}" exec bash "${SCRIPT_DIR}/run_train.sh" "${ARM}" "${STEPS}" "${TAG}"
