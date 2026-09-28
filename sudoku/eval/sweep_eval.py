#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep evaluation across all hyperparameter combinations.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def get_absorbing_ratios():
    """Standard absorbing mask ratios."""
    # return [0.3,0.4,0.5,0.6,0.7]
    return [0.3, 0.4, 0.5, 0.6]


def get_uniform_ratios():
    """Standard uniform noise ratios."""
    head = [0.3, 0.2, 0.1]
    # tail = [0.08, 0.06, 0.04, 0.02]
    tail = []

    return sorted(head + tail, reverse=True)


def get_steps():
    """Standard diffusion steps."""
    return list(range(1, 17))


def get_algorithm_configs():
    """Return algorithm configurations as tuples (algorithm, param_name, param_value).
    
    Returns:
        List of tuples:
        - ("random-remask", None, None): regular random remask
        - ("self_conf-remask:vanilla", "confidence_threshold", value): vanilla with threshold
        - ("self_conf-remask:vanilla_MAD", "mad_k", value): MAD with k value
    """
    configs = [
        # Regular algorithms
        ("random-remask", None, None),
        
        # self_conf-remask:vanilla with confidence thresholds
        ("self_conf-remask:vanilla", "confidence_threshold", -1),
        ("self_conf-remask:vanilla", "confidence_threshold", 0.95),
        ("self_conf-remask:vanilla", "confidence_threshold", 0.9),
        ("self_conf-remask:vanilla", "confidence_threshold", 0.8),
        ("self_conf-remask:vanilla", "confidence_threshold", 0.7),
        
        # self_conf-remask:vanilla_MAD with mad_k values
        # ("self_conf-remask:vanilla_MAD", "mad_k", 2.5),
        # ("self_conf-remask:vanilla_MAD", "mad_k", 2.0),
        # ("self_conf-remask:vanilla_MAD", "mad_k", 3.0),
    ]
    return configs


def get_uniform_noise_diffusion_configs():
    """Return (noise_ratio, editable_ratio) configurations for uniform_noise_diffusion mode.
    
    Returns:
        List of tuples: [(noise_ratio, editable_ratio), ...]
    """
    configs = [
        (0.3, 0.4),
        (0.3, 0.5),
        (0.3, 0.6),
        (0.2, 0.4),
        (0.2, 0.5),
        (0.2, 0.6),
        (0.1, 0.4),
        (0.1, 0.5),
        (0.1, 0.6),
    ]
    return configs


def compute_total_evaluations(
    modes,
    absorbing_ratios,
    uniform_ratios,
    steps,
    algorithm_configs,
    uniform_noise_diffusion_configs,
):
    total = 0
    if "absorbing" in modes:
        total += len(absorbing_ratios) * len(algorithm_configs) * len(steps)
    if "uniform_noise_only" in modes:
        total += len(uniform_ratios)
    if "uniform_noise_diffusion" in modes:
        total += len(uniform_noise_diffusion_configs) * len(algorithm_configs) * len(steps)
    return total


def run_eval(
    checkpoint: str,
    data_dir: str,
    eval_mode: str,
    device: str,
    project: str,
    eval_batch_size: int,
    num_samples: int,
    **kwargs
):
    """Run a single evaluation and return metrics."""
    cmd = [
        "python",
        "eval/run_all_evals.py",
        "--checkpoint", checkpoint,
        "--data-dir", data_dir,
        "--num-samples", str(num_samples),
        "--eval-mode", eval_mode,
        "--device", device,
        "--project", project,
        "--eval-batch-size", str(eval_batch_size),
    ]

    # Add mode-specific arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=ROOT_DIR, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(
            f"Warning: Evaluation failed with return code "
            f"{result.returncode}"
        )
        print(f"stderr: {result.stderr}")

    # Parse metrics from stdout
    metrics = {"eval_mode": eval_mode, **kwargs}
    for line in result.stdout.splitlines():
        if line.startswith("EVAL_METRICS_JSON="):
            payload = line.split("=", 1)[1]
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    metrics.update(parsed)
            except Exception:
                import pdb; pdb.set_trace()
    return result.returncode, metrics


def save_results(output_dir: Path, mode: str, entries: list):
    """Save evaluation results for a specific mode."""
    if not entries:
        return
    
    json_file = output_dir / f"{mode}_results.json"
    with open(json_file, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"\nSaved JSON results to: {json_file}")

    csv_file = output_dir / f"{mode}_results.csv"
    fieldnames = sorted({key for item in entries for key in item.keys()})
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in entries:
            row = {name: item.get(name, "") for name in fieldnames}
            writer.writerow(row)
    print(f"Saved CSV results to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep evaluation across all hyperparameter combinations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sudoku_eval",
        help="wandb project name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2000,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "absorbing",
            "uniform_noise_only",
            "uniform_noise_diffusion"
        ],
        choices=[
            "absorbing",
            "uniform_noise_only",
            "uniform_noise_diffusion"
        ],
        help="Evaluation modes to run",
    )

    args = parser.parse_args()

    # Prepare output directory
    # NOTE: This script must be run from the sudoku/ directory (parent of eval/)
    # so that checkpoint paths can be correctly resolved as relative paths.
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    rel_parts = checkpoint_path.relative_to(Path.cwd()).parts
    output_dir = base_output_dir.joinpath(
        *rel_parts[:-1],
        checkpoint_path.stem,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use built-in functions to get parameters
    absorbing_ratios = get_absorbing_ratios()
    uniform_ratios = get_uniform_ratios()
    steps = get_steps()
    algorithm_configs = get_algorithm_configs()
    uniform_noise_diffusion_configs = get_uniform_noise_diffusion_configs()

    total_runs = 0
    failed_runs = 0
    results_by_mode = {
        "absorbing": [],
        "uniform_noise_only": [],
        "uniform_noise_diffusion": [],
    }

    # Compute expected total evaluations for global progress bar
    expected_total = compute_total_evaluations(
        args.modes,
        absorbing_ratios,
        uniform_ratios,
        steps,
        algorithm_configs,
        uniform_noise_diffusion_configs,
    )
    overall_pbar = tqdm(total=expected_total, desc="Total evals")

    # 1. Absorbing evaluation
    if "absorbing" in args.modes:
        print("\n" + "="*80)
        print("ABSORBING EVALUATION")
        print("="*80)
        for mask_ratio in tqdm(absorbing_ratios, desc="Absorbing ratios"):
            for algo_config in tqdm(
                algorithm_configs,
                desc="Algorithm configs",
                leave=False,
            ):
                algorithm, param_name, param_value = algo_config
                for step in tqdm(steps, desc="Steps", leave=False):
                    total_runs += 1
                    eval_kwargs = {
                        "checkpoint": args.checkpoint,
                        "data_dir": args.data_dir,
                        "eval_mode": "absorbing",
                        "device": args.device,
                        "project": args.project,
                        "eval_batch_size": args.eval_batch_size,
                        "num_samples": args.num_samples,
                        "mask_ratio": mask_ratio,
                        "algorithm": algorithm,
                        "step": step,
                    }
                    # Add parameter if specified
                    if param_name is not None and param_value is not None:
                        eval_kwargs[param_name] = param_value
                    
                    ret, metrics = run_eval(**eval_kwargs)
                    results_by_mode["absorbing"].append(metrics)
                    if ret != 0:
                        failed_runs += 1
                    overall_pbar.update(1)
        
        # Save absorbing results immediately after completion
        save_results(output_dir, "absorbing", results_by_mode["absorbing"])

    # 2. Uniform noise only evaluation
    if "uniform_noise_only" in args.modes:
        print("\n" + "="*80)
        print("UNIFORM NOISE ONLY EVALUATION")
        print("="*80)
        for noise_ratio in tqdm(uniform_ratios, desc="Uniform ratios"):
            total_runs += 1
            ret, metrics = run_eval(
                checkpoint=args.checkpoint,
                data_dir=args.data_dir,
                eval_mode="uniform_noise_only",
                device=args.device,
                project=args.project,
                eval_batch_size=args.eval_batch_size,
                num_samples=args.num_samples,
                noise_ratio=noise_ratio,
            )
            results_by_mode["uniform_noise_only"].append(metrics)
            if ret != 0:
                failed_runs += 1
            overall_pbar.update(1)
        
        # Save uniform_noise_only results immediately after completion
        save_results(output_dir, "uniform_noise_only", results_by_mode["uniform_noise_only"])

    # 3. Uniform noise + diffusion evaluation
    if "uniform_noise_diffusion" in args.modes:
        print("\n" + "="*80)
        print("UNIFORM NOISE + DIFFUSION EVALUATION")
        print("="*80)
        mode_key = "uniform_noise_diffusion"
        for noise_ratio, editable_ratio in tqdm(
            uniform_noise_diffusion_configs,
            desc="Noise/Editable ratios",
        ):
            for algo_config in tqdm(
                algorithm_configs,
                desc="Algorithm configs",
                leave=False,
            ):
                algorithm, param_name, param_value = algo_config
                for step in tqdm(steps, desc="Steps", leave=False):
                    total_runs += 1
                    eval_kwargs = {
                        "checkpoint": args.checkpoint,
                        "data_dir": args.data_dir,
                        "eval_mode": "uniform_noise_diffusion",
                        "device": args.device,
                        "project": args.project,
                        "eval_batch_size": args.eval_batch_size,
                        "num_samples": args.num_samples,
                        "noise_ratio": noise_ratio,
                        "editable_ratio": editable_ratio,
                        "algorithm": algorithm,
                        "step": step,
                    }
                    # Add parameter if specified
                    if param_name is not None and param_value is not None:
                        eval_kwargs[param_name] = param_value
                    
                    ret, metrics = run_eval(**eval_kwargs)
                    results_by_mode[mode_key].append(metrics)
                    if ret != 0:
                        failed_runs += 1
                    overall_pbar.update(1)
        
        # Save uniform_noise_diffusion results immediately after completion
        save_results(output_dir, mode_key, results_by_mode[mode_key])

    overall_pbar.close()

    print("\n" + "="*80)
    print(f"SWEEP COMPLETE: {total_runs} total runs, {failed_runs} failed")
    print("="*80)


if __name__ == "__main__":
    main()
