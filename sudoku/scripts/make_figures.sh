#!/usr/bin/env bash
# Render the Appendix F figures from the sweep results into FIG_DIR.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FIG_DIR="$(_abspath "${FIG_DIR:-${RESULTS_ROOT}/figures}")"
MDLM="checkpoints_absorbing/${CHECKPOINT_STEM}"
CDLM="uniform_absorbing_mixture_checkpoints/prob_${MIXTURE_PROB}/${CHECKPOINT_STEM}"
REFINEMENT="${RESULTS_ROOT}/refinement"
COMPLETION="${RESULTS_ROOT}/completion"

for f in "${REFINEMENT}/${MDLM}/uniform_noise_only_results.csv" \
         "${REFINEMENT}/${CDLM}/uniform_noise_only_results.csv" \
         "${REFINEMENT}/${MDLM}/uniform_noise_diffusion_results.csv" \
         "${REFINEMENT}/${CDLM}/uniform_noise_diffusion_results.csv" \
         "${COMPLETION}/${MDLM}/absorbing_results.csv" \
         "${COMPLETION}/${CDLM}/absorbing_results.csv"; do
    if [ ! -f "${f}" ]; then
        echo "Results not found: ${f} (run scripts/eval_refinement.sh and scripts/eval_completion.sh)" >&2
        exit 1
    fi
done

cd "${SUDOKU_DIR}/eval/analysis"

# Confidence on clean vs corrupted cells -> sudoku_uniform_noise_comparison.pdf
python compare_all.py \
    --result-dirs "${REFINEMENT}/${MDLM}" "${REFINEMENT}/${CDLM}" \
    --output-dir "${FIG_DIR}"

# Editable-region refinement -> uniform_noise_diffusion_compare/
#   (the paper shows ..._confidence_threshold_08_comparison.pdf)
python compare_all_line_plots.py \
    --result-dirs "${REFINEMENT}/${MDLM}" "${REFINEMENT}/${CDLM}" \
    --output-dir "${FIG_DIR}" \
    --combinations 0.1,0.4 0.1,0.5 0.1,0.6 0.2,0.4 0.2,0.5 0.2,0.6

# Pure completion -> absorbing_completion_compare/
#   (the paper shows ..._confidence_threshold_07_comparison.pdf)
python compare_completion_line_plots.py \
    --result-dirs "${COMPLETION}/${MDLM}" "${COMPLETION}/${CDLM}" \
    --output-dir "${FIG_DIR}" \
    --mask-ratios 0.3 0.4 0.5 0.6 \
    --steps 1 2 3 4 5 6 7 8
