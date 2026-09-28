#!/usr/bin/env bash
# Open-dCoder-0.5B continued training on nvidia/OpenCodeInstruct, the ungated substitute for the
# paper's Nemotron-SFT-Code corpus (see training/data_prep/README.md). Runs made with this script
# are not the paper's runs and do not reproduce its numbers.
#
# This is train_0.5b.sh with TRAIN_PATH taken from the train_path.txt written by
# training/data_prep/prepare_opencodeinstruct.py. Everything else is unchanged: the CDLM-0.5B
# defaults (ARM=cdlm, SEED=42, long-horizon LR schedule, stop and save at STOP_STEP=2000) and all
# environment variables of train_0.5b.sh, with the same defaults, except:
#
#   OCI_TRAIN_PATH_FILE   train_path.txt of the rendered shards
#                         [${DATA_ROOT:-<repo>/data}/OpenCodeInstruct-text/train_path.txt]
#   RUN_NAME              run directory name  [open-dcoder-0.5B-oci-<ARM>-seed<SEED>-step<STOP_STEP>]
#
# TRAIN_PATH is always set from OCI_TRAIN_PATH_FILE, so DATA_DIR and PAPER_ORDER have no effect.
#
# Usage:
#   python training/data_prep/prepare_opencodeinstruct.py      # once
#   bash training/scripts/train_0.5b_opencodeinstruct.sh
#   ARM=mdlm SEED=1234 NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash training/scripts/train_0.5b_opencodeinstruct.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

export ARM="${ARM:-cdlm}"
export SEED="${SEED:-42}"
export STOP_STEP="${STOP_STEP:-2000}"
export RUN_NAME="${RUN_NAME:-open-dcoder-0.5B-oci-${ARM}-seed${SEED}-step${STOP_STEP}}"

OCI_TRAIN_PATH_FILE="${OCI_TRAIN_PATH_FILE:-${DATA_ROOT:-${REPO_ROOT}/data}/OpenCodeInstruct-text/train_path.txt}"
[[ -s "${OCI_TRAIN_PATH_FILE}" ]] || die "missing ${OCI_TRAIN_PATH_FILE}; build it with
  python ${REPO_ROOT}/training/data_prep/prepare_opencodeinstruct.py [--local_dir DATA_ROOT]
or set OCI_TRAIN_PATH_FILE to its train_path.txt"

oci_train_path="$(tr -d '\n' < "${OCI_TRAIN_PATH_FILE}")"
[[ -n "${oci_train_path}" ]] || die "${OCI_TRAIN_PATH_FILE} is empty"
if [[ -n "${TRAIN_PATH:-}" && "${TRAIN_PATH}" != "${oci_train_path}" ]]; then
    echo "NOTE: ignoring the TRAIN_PATH from the environment; using ${OCI_TRAIN_PATH_FILE}" >&2
fi
export TRAIN_PATH="${oci_train_path}"

echo "Training data:      nvidia/OpenCodeInstruct, rendered (${OCI_TRAIN_PATH_FILE})"
exec bash "${SCRIPT_DIR}/train_0.5b.sh"
