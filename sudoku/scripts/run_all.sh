#!/usr/bin/env bash
# Full Appendix F pipeline: data -> train MDLM and CDLM -> sweeps -> figures.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPTS}/generate_data.sh"
bash "${SCRIPTS}/train.sh" absorbing
bash "${SCRIPTS}/train.sh" mixture
bash "${SCRIPTS}/eval_refinement.sh"
bash "${SCRIPTS}/eval_completion.sh"
bash "${SCRIPTS}/make_figures.sh"
