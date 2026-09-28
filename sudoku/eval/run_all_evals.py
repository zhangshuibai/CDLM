#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single evaluation run with specified mode.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval.absorbing_eval import evaluate_absorbing  # noqa: E402
from eval.uniform_noise_diffusion_eval import (  # noqa: E402
    evaluate_uniform_noise_with_diffusion,
)
from eval.uniform_noise_eval import evaluate_uniform_noise_only  # noqa: E402
from eval.utils import (  # noqa: E402
    init_wandb_run,
    load_model_and_data,
)


def collect_run_config(resources, project: str, args) -> dict:
    config = dict(resources.checkpoint_meta)
    config["project"] = project
    config["eval_mode"] = args.eval_mode
    if isinstance(resources.checkpoint_meta.get("config"), dict):
        for key, value in resources.checkpoint_meta["config"].items():
            config[f"checkpoint_config/{key}"] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single evaluation mode",
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
        "--num-samples",
        type=int,
        default=2000,
        help="Number of test samples",
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
        "--run-name",
        type=str,
        default=None,
        help="wandb run name",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        required=True,
        choices=["absorbing", "uniform_noise_only", "uniform_noise_diffusion"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2000,
        help="Batch size for evaluation",
    )

    # Absorbing-specific
    parser.add_argument(
        "--mask-ratio",
        type=float,
        help="Mask ratio for absorbing eval",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Diffusion steps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        help="Sampling algorithm",
    )
    parser.add_argument(
        "--remask-scheduler",
        type=str,
        default="linear",
        help="Remask scheduler",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for remasking",
    )
    parser.add_argument(
        "--mad-k",
        type=float,
        default=None,
        help="MAD k parameter for self_conf-remask:vanilla_MAD algorithm",
    )

    # Uniform noise specific
    parser.add_argument(
        "--noise-ratio",
        type=float,
        help="Noise ratio for uniform evals",
    )

    # Uniform noise + diffusion specific
    parser.add_argument(
        "--editable-ratio",
        type=float,
        help="Editable ratio for uniform+diffusion eval",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resources = load_model_and_data(
        args.checkpoint,
        args.data_dir,
        device,
        args.num_samples,
    )

    config = collect_run_config(resources, args.project, args)

    if args.eval_mode == "absorbing":
        if (
            args.mask_ratio is None
            or args.step is None
            or args.algorithm is None
        ):
            raise ValueError(
                "absorbing mode requires --mask-ratio, --step, --algorithm"
            )
        config.update({
            "mask_ratio": args.mask_ratio,
            "step": args.step,
            "algorithm": args.algorithm,
            "remask_scheduler": args.remask_scheduler,
        })
        if args.confidence_threshold is not None:
            config["confidence_threshold"] = args.confidence_threshold
        if args.mad_k is not None:
            config["mad_k"] = args.mad_k
        init_wandb_run(args.project, args.run_name, config)
        metrics = evaluate_absorbing(
            resources=resources,
            mask_ratio=args.mask_ratio,
            step=args.step,
            algorithm=args.algorithm,
            remask_scheduler=args.remask_scheduler,
            eval_batch_size=args.eval_batch_size,
            confidence_threshold=args.confidence_threshold,
            mad_k=args.mad_k,
        )
        print("EVAL_METRICS_JSON=" + json.dumps(metrics, ensure_ascii=False))

    elif args.eval_mode == "uniform_noise_only":
        if args.noise_ratio is None:
            raise ValueError("uniform_noise_only mode requires --noise-ratio")
        config.update({
            "noise_ratio": args.noise_ratio,
        })
        init_wandb_run(args.project, args.run_name, config)
        metrics = evaluate_uniform_noise_only(
            resources=resources,
            noise_ratio=args.noise_ratio,
            eval_batch_size=args.eval_batch_size,
        )
        print("EVAL_METRICS_JSON=" + json.dumps(metrics, ensure_ascii=False))

    elif args.eval_mode == "uniform_noise_diffusion":
        if (
            args.noise_ratio is None
            or args.editable_ratio is None
            or args.step is None
            or args.algorithm is None
        ):
            raise ValueError(
                "uniform_noise_diffusion mode requires --noise-ratio, "
                "--editable-ratio, --step, --algorithm"
            )
        config.update({
            "noise_ratio": args.noise_ratio,
            "editable_ratio": args.editable_ratio,
            "step": args.step,
            "algorithm": args.algorithm,
            "remask_scheduler": args.remask_scheduler,
        })
        if args.confidence_threshold is not None:
            config["confidence_threshold"] = args.confidence_threshold
        if args.mad_k is not None:
            config["mad_k"] = args.mad_k
        init_wandb_run(args.project, args.run_name, config)
        metrics = evaluate_uniform_noise_with_diffusion(
            resources=resources,
            noise_ratio=args.noise_ratio,
            editable_ratio=args.editable_ratio,
            step=args.step,
            algorithm=args.algorithm,
            remask_scheduler=args.remask_scheduler,
            eval_batch_size=args.eval_batch_size,
            confidence_threshold=args.confidence_threshold,
            mad_k=args.mad_k,
        )
        print("EVAL_METRICS_JSON=" + json.dumps(metrics, ensure_ascii=False))

    print(f"Evaluation complete: {args.eval_mode}")


if __name__ == "__main__":
    main()
