#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run eval/sweep_eval.py with the grid behind a given appendix figure.

sweep_eval.py is kept exactly as committed; its built-in grid is the one used for
the completion sweep. The refinement sweep was run with an earlier revision of
sweep_eval.py that differed only in get_steps() and get_algorithm_configs(); the
"refinement" preset restores those two functions and changes nothing else.

Usage: same arguments as sweep_eval.py, plus --preset {refinement,completion}.
"""

import argparse
import sys

import sweep_eval

PRESETS = {
    # sudoku_uniform_noise_comparison.pdf and
    # sudoku_uniform_noise_diffusion_*_comparison.pdf
    "refinement": {
        "steps": [1, 2, 3, 4],
        "algorithm_configs": [
            ("random-remask", None, None),
            ("self_conf-remask:vanilla", "confidence_threshold", -1),
            ("self_conf-remask:vanilla", "confidence_threshold", 0.95),
            ("self_conf-remask:vanilla", "confidence_threshold", 0.9),
            ("self_conf-remask:vanilla", "confidence_threshold", 0.8),
            ("self_conf-remask:vanilla_MAD", "mad_k", 2.5),
            ("self_conf-remask:vanilla_MAD", "mad_k", 2.0),
            ("self_conf-remask:vanilla_MAD", "mad_k", 3.0),
        ],
    },
    # sudoku_absorbing_*_comparison.pdf: the grid built into sweep_eval.py
    # (steps 1-16; random-remask and thresholds -1, 0.95, 0.9, 0.8, 0.7).
    "completion": None,
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    args, rest = parser.parse_known_args()

    preset = PRESETS[args.preset]
    if preset is not None:
        sweep_eval.get_steps = lambda: list(preset["steps"])
        sweep_eval.get_algorithm_configs = lambda: list(preset["algorithm_configs"])

    sys.argv = [sys.argv[0]] + rest
    sweep_eval.main()


if __name__ == "__main__":
    main()
