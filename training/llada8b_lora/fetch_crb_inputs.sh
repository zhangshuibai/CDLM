#!/bin/bash
# Download the 24 CRB input files of the LLaDA-8B LoRA experiment into <repo>/buggy_datasets/.
#   usage: bash fetch_crb_inputs.sh [--force]
#
# Source: the Hugging Face dataset Shuibai12138/crb-paper-inputs at a pinned revision.
#   llada-8b-base/<ds>/evaluated/LLaDA-8B-Base_<err>_2_wrong_1_evaluated.jsonl
#       -> buggy_datasets/<ds>/evaluated/...   read by the CRB repair sweep (run_crb_cell.sh)
#   llada-8b-base-localisation/<ds>/LLaDA-8B-Base_<err>_2_wrong_1.jsonl
#       -> buggy_datasets/<ds>/...             read by the localisation evaluation (eval_confidence.py)
# For var / literal, the two sets hold partly different instances (see the dataset card); this is how the
# experiment was run. Every file is checked against crb_inputs.md5. An existing file with a different
# md5 is an error unless --force is given, which replaces it.
#
# Environment overrides: CRB_INPUTS_REVISION [pinned below], PYTHON [python].
set -euo pipefail

REVISION=${CRB_INPUTS_REVISION:-a00037635943c930fa5d8e0961c7ca26127ca53f}
FORCE=0
case "${1:-}" in --force) FORCE=1 ;; "") ;; *) echo "usage: fetch_crb_inputs.sh [--force]"; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-python}
cd "${REPO_ROOT}"

TMP="$(mktemp -d "${REPO_ROOT}/.crb_inputs_download.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

"${PY}" - "${REVISION}" "${TMP}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
rev, out = sys.argv[1], sys.argv[2]
snapshot_download("Shuibai12138/crb-paper-inputs", repo_type="dataset", revision=rev, local_dir=out,
                  allow_patterns=["llada-8b-base/*/evaluated/LLaDA-8B-Base_*_2_wrong_1_evaluated.jsonl",
                                  "llada-8b-base-localisation/*/LLaDA-8B-Base_*_2_wrong_1.jsonl"])
EOF

N=0
while read -r sum path; do
    rel="${path#buggy_datasets/}"
    case "${rel}" in
        */evaluated/*) src="${TMP}/llada-8b-base/${rel}" ;;
        *)             src="${TMP}/llada-8b-base-localisation/${rel}" ;;
    esac
    [ -f "${src}" ] || { echo "missing in the download: ${src#"${TMP}"/}"; exit 1; }
    got="$(md5sum < "${src}" | cut -d' ' -f1)"
    [ "${got}" = "${sum}" ] || { echo "md5 mismatch for downloaded ${src#"${TMP}"/}"; exit 1; }
    if [ -e "${path}" ]; then
        have="$(md5sum < "${path}" | cut -d' ' -f1)"
        if [ "${have}" = "${sum}" ]; then N=$((N + 1)); continue; fi
        if [ "${FORCE}" != 1 ]; then
            echo "${path} exists with a different md5; move it away or pass --force"; exit 1
        fi
    fi
    mkdir -p "$(dirname "${path}")"
    cp "${src}" "${path}"
    N=$((N + 1))
done < "${SCRIPT_DIR}/crb_inputs.md5"

md5sum -c --quiet "${SCRIPT_DIR}/crb_inputs.md5"
echo "OK: ${N} CRB input files in ${REPO_ROOT}/buggy_datasets (Shuibai12138/crb-paper-inputs@${REVISION})"
