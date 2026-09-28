# Shared defaults for the Sudoku scripts; sourced, not executed.
# Every variable below can be overridden from the environment.

_abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${PWD}/$1" ;; esac; }

SUDOKU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$(_abspath "${DATA_DIR:-${SUDOKU_DIR}/data}")"
CKPT_ROOT="$(_abspath "${CKPT_ROOT:-${SUDOKU_DIR}/checkpoints}")"
RESULTS_ROOT="$(_abspath "${RESULTS_ROOT:-${SUDOKU_DIR}/results}")"

# Uniform replacement probability of the mixture (CDLM) model; the paper uses 0.1.
# Write it the way Python prints the float (0.1, 0.02, 0.2): train.py names the
# checkpoint directory prob_<value>.
MIXTURE_PROB="${MIXTURE_PROB:-0.1}"

# DataLoader workers for training (the paper runs used 64; this only affects
# loading throughput, not the sampled data order).
NUM_WORKERS="${NUM_WORKERS:-8}"

# Checkpoint evaluated and plotted; the paper uses best_model.pt.
CHECKPOINT="${CHECKPOINT:-best_model.pt}"
# sweep_eval.py names the result directory after the checkpoint without suffix.
CHECKPOINT_STEM="${CHECKPOINT%.*}"
DEVICE="${DEVICE:-cuda}"

# Logging is off unless WANDB_MODE is set (e.g. WANDB_MODE=online).
export WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-sudoku-diffusion-lm}"
WANDB_PROJECT_EVAL="${WANDB_PROJECT_EVAL:-sudoku_eval}"

# These directory names are load-bearing: the plotting scripts label a run
# "MDLM" when its path contains "checkpoints_absorbing" and "CDLM" when it
# contains "prob_".
ABSORBING_CKPT_DIR="${CKPT_ROOT}/checkpoints_absorbing"
MIXTURE_CKPT_DIR="${CKPT_ROOT}/uniform_absorbing_mixture_checkpoints/prob_${MIXTURE_PROB}"

# check_sweep <log> <result_dir> <mode>...
# sweep_eval.py exits 0 even when evaluation points fail, so check its summary
# line and require the metric column to be filled on every row of each CSV.
_CHECK_CSV='
import csv, sys
path, col = sys.argv[1:]
try:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
except OSError as e:
    sys.exit(f"Missing result file: {e}")
bad = sum(1 for r in rows if not r.get(col))
if not rows or bad:
    sys.exit(f"{path}: {bad} of {len(rows)} rows have no {col}")
'
check_sweep() {
    local log="$1" dir="$2" summary failed mode col
    shift 2
    summary="$(grep -E '^SWEEP COMPLETE: [0-9]+ total runs, [0-9]+ failed$' "${log}" | tail -n 1 || true)"
    if [ -z "${summary}" ]; then
        echo "Sweep did not finish: no summary line in ${log}" >&2
        return 1
    fi
    failed="${summary##*, }"
    failed="${failed% failed}"
    if [ "${failed}" -ne 0 ]; then
        echo "${summary} (see ${log})" >&2
        return 1
    fi
    for mode in "$@"; do
        case "${mode}" in
            uniform_noise_only) col=clean_token_confidence ;;
            *) col=board_accuracy ;;
        esac
        python -c "${_CHECK_CSV}" "${dir}/${mode}_results.csv" "${col}" || return 1
    done
}

# run_sweep <preset> <output_dir> <modes>
# Runs eval/sweep_eval.py (through eval/sweep_presets.py) on the MDLM and CDLM
# checkpoints. sweep_eval.py names its output directory after the checkpoint path
# relative to the current directory, so it is launched from CKPT_ROOT; results
# land in <output_dir>/checkpoints_absorbing/<stem>/ and
# <output_dir>/uniform_absorbing_mixture_checkpoints/prob_<p>/<stem>/, together
# with the sweep's stdout in <preset>_sweep.log. Exits non-zero if any point failed.
run_sweep() {
    local preset="$1" out="$2" modes="$3" ckpt_dir ckpt root result_dir log
    for ckpt_dir in "${ABSORBING_CKPT_DIR}" "${MIXTURE_CKPT_DIR}"; do
        if [ ! -f "${ckpt_dir}/${CHECKPOINT}" ]; then
            echo "Checkpoint not found: ${ckpt_dir}/${CHECKPOINT}" >&2
            if [ "${CHECKPOINT}" = best_model.pt ]; then
                echo "train.py writes best_model.pt only when the held-out score improves; for short runs set CHECKPOINT=final_model.pt" >&2
            fi
            exit 1
        fi
    done
    root="$(cd "${CKPT_ROOT}" && pwd -P)"
    mkdir -p "${out}"
    for ckpt_dir in "${ABSORBING_CKPT_DIR}" "${MIXTURE_CKPT_DIR}"; do
        ckpt="$(cd "${ckpt_dir}" && pwd -P)/${CHECKPOINT}"
        result_dir="${out}/${ckpt_dir#"${CKPT_ROOT}/"}/${CHECKPOINT_STEM}"
        log="${result_dir}/${preset}_sweep.log"
        mkdir -p "${result_dir}"
        echo "=== ${preset} sweep: ${ckpt}"
        # stdin is /dev/null so that the pdb.set_trace() in sweep_eval.py's
        # metrics-parsing error path aborts the sweep instead of waiting for input.
        # shellcheck disable=SC2086
        if ! (cd "${root}" && python -u "${SUDOKU_DIR}/eval/sweep_presets.py" \
            --preset "${preset}" \
            --checkpoint "${ckpt}" \
            --data-dir "${DATA_DIR}" \
            --device "${DEVICE}" \
            --project "${WANDB_PROJECT_EVAL}" \
            --output-dir "${out}" \
            --eval-batch-size 2000 \
            --num-samples 2000 \
            --modes ${modes} < /dev/null) | tee "${log}"; then
            echo "Sweep aborted: ${ckpt} (see ${log})" >&2
            exit 1
        fi
        # shellcheck disable=SC2086
        if ! check_sweep "${log}" "${result_dir}" ${modes}; then
            echo "Sweep failed: ${ckpt} (see ${log})" >&2
            exit 1
        fi
    done
}
