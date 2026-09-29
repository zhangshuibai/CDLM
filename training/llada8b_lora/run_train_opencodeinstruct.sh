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
# What a run reads: rank r reads the shards files[r::NPROC] of that shuffled list one after the
# other, from their first row, one document per sequence, and its two DataLoader workers each yield
# every document (README.md, "Data loader"). A 2000-step run with 4 ranks therefore reads about
# 3,000 documents from the head of one shard per rank. OpenCodeInstruct's shards are not mixed by
# category, so this is not a sample of the whole corpus. data_shards.tsv in the run directory
# records every shard and the rank and position at which it is read.
#
# Environment overrides (defaults in brackets), in addition to those of run_train.sh:
#   OCI_TRAIN_PATH_FILE [<repo>/data/OpenCodeInstruct-text/train_path.txt]
#   OCI_FLAT_DIR        [<directory of OCI_TRAIN_PATH_FILE>-flat, e.g. data/OpenCodeInstruct-text-flat]
#   OCI_ALLOW_SUBSET    [0]  set to 1 to train on a --shards subset (smoke tests only); it needs at
#                            least one shard per rank (--shards >= NPROC, or NPROC=1)
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
NP=${NPROC:-4}     # run_train.sh's default
SEED=42            # run_train.sh passes --seed 42

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
# A rank without a shard never yields a batch and the run hangs, so refuse that here.
if [ "${#DIRS[@]}" -lt "${NP}" ]; then
    echo "${#DIRS[@]} shard(s) for ${NP} ranks: every rank needs one (render --shards ${NP}, or set NPROC=1)"
    exit 1
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
    name="$(basename -- "${files[0]}")"
    WANT+=("${name}")
    link="${FLAT}/${name}"
    # Never re-point an existing link: another run may be reading this directory.
    if [ -L "${link}" ] || [ -e "${link}" ]; then
        if [ "$(readlink -- "${link}" 2>/dev/null)" != "${src}" ]; then
            echo "${link} exists and does not link to ${src}; remove it or set OCI_FLAT_DIR"
            exit 1
        fi
    elif ! ln -s -- "${src}" "${link}" 2>/dev/null && [ "$(readlink -- "${link}")" != "${src}" ]; then
        echo "cannot link ${link} -> ${src}"
        exit 1
    fi
done
if [ "$(printf '%s\n' "${WANT[@]}" | sort | uniq -d)" != "" ]; then
    echo "${LIST} lists two shards with the same file name"
    exit 1
fi
# The trainer reads every *.parquet file in the directory, so nothing else may be there.
EXTRA="$(comm -23 <(cd "${FLAT}" && find . -maxdepth 1 -name '*.parquet' -printf '%f\n' | sort) \
                  <(printf '%s\n' "${WANT[@]}" | sort))"
if [ -n "${EXTRA}" ]; then
    echo "${FLAT} holds .parquet files that ${LIST} does not list:" ${EXTRA}
    exit 1
fi

# Record the shards and the rank and position at which each is read (the trainer's order).
OUT=${OUTPUT_DIR:-${REPO_ROOT}/outputs/llada8b_lora}/${TAG}_${ARM}
mkdir -p "${OUT}"
"${PY}" - "${FLAT}" "${OUT}/data_shards.tsv" "${NP}" "${SEED}" "${STEPS}" <<'EOF'
import json, os, random, sys
import pyarrow.parquet as pq
flat, out, world, seed, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
names = sorted(n for n in os.listdir(flat) if n.endswith(".parquet"))
order = list(names)                      # train_llada_mixture.PackedParquetStream: sorted paths,
random.Random(seed).shuffle(order)       # then random.Random(seed).shuffle, then [rank::world]
where = {n: (i % world, i // world) for i, n in enumerate(order)}
with open(out, "w") as f:
    f.write("file\trows\trank\trank_position\tsource\trevision\tsource_sha256\ttemplate\n")
    for name in names:
        pf = pq.ParquetFile(os.path.join(flat, name))
        info = json.loads(pf.schema_arrow.metadata[b"cdlm_render"])
        rank, pos = where[name]
        f.write("\t".join([name, str(pf.metadata.num_rows), str(rank), str(pos), info["source"], info["revision"],
                           info["source_sha256"], json.dumps(info["template"])]) + "\n")
per_rank = steps * (12 // world) // 2    # micro 1 x accum (12 / world) per step, each document twice
first = [order[r] for r in range(world)]
print(f"each rank reads about {per_rank:,} documents (one per sequence, documents of at most 4,096 tokens), "
      f"from the head of its first shard: " + ", ".join(f"rank {r} {n}" for r, n in enumerate(first)))
EOF
echo "OpenCodeInstruct: ${#WANT[@]} rendered shards linked in ${FLAT} (listed in ${OUT}/data_shards.tsv)"

DATA_DIR="${FLAT}" exec bash "${SCRIPT_DIR}/run_train.sh" "${ARM}" "${STEPS}" "${TAG}"
