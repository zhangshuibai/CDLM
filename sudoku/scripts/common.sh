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

# run_sweep <preset> <output_dir> <modes>
# Runs eval/sweep_eval.py (through eval/sweep_presets.py) on the MDLM and CDLM
# checkpoints. sweep_eval.py names its output directory after the checkpoint path
# relative to the current directory, so it is launched from CKPT_ROOT; results
# land in <output_dir>/checkpoints_absorbing/<stem>/ and
# <output_dir>/uniform_absorbing_mixture_checkpoints/prob_<p>/<stem>/.
run_sweep() {
    local preset="$1" out="$2" modes="$3" ckpt_dir ckpt root
    for ckpt_dir in "${ABSORBING_CKPT_DIR}" "${MIXTURE_CKPT_DIR}"; do
        if [ ! -f "${ckpt_dir}/${CHECKPOINT}" ]; then
            echo "Checkpoint not found: ${ckpt_dir}/${CHECKPOINT}" >&2
            exit 1
        fi
    done
    root="$(cd "${CKPT_ROOT}" && pwd -P)"
    mkdir -p "${out}"
    for ckpt_dir in "${ABSORBING_CKPT_DIR}" "${MIXTURE_CKPT_DIR}"; do
        ckpt="$(cd "${ckpt_dir}" && pwd -P)/${CHECKPOINT}"
        echo "=== ${preset} sweep: ${ckpt}"
        # shellcheck disable=SC2086
        (cd "${root}" && python "${SUDOKU_DIR}/eval/sweep_presets.py" \
            --preset "${preset}" \
            --checkpoint "${ckpt}" \
            --data-dir "${DATA_DIR}" \
            --device "${DEVICE}" \
            --project "${WANDB_PROJECT_EVAL}" \
            --output-dir "${out}" \
            --eval-batch-size 2000 \
            --num-samples 2000 \
            --modes ${modes})
    done
}
