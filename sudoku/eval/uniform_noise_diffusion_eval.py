#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation under uniform replacement noise with diffusion-based correction.
"""

import argparse
import sys
from pathlib import Path

import torch
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval.utils import (  # noqa: E402
    EvaluationResources,
    apply_uniform_noise,
    build_editable_mask,
    compute_board_accuracy,
    compute_masked_loss_and_accuracy,
    forward_in_batches,
    load_model_and_data,
    log_metrics,
    run_diffusion_sampling,
)


def evaluate_uniform_noise_with_diffusion(
    resources: EvaluationResources,
    noise_ratio: float,
    editable_ratio: float,
    step: int,
    algorithm: str,
    remask_scheduler: str,
    eval_batch_size: int | None,
    confidence_threshold: float | None = None,
    mad_k: float | None = None,
) -> dict:
    if editable_ratio < noise_ratio:
        raise ValueError(
            f"editable_ratio ({editable_ratio}) must be >= "
            f"noise_ratio ({noise_ratio})"
        )

    model = resources.model
    device = resources.device
    targets = resources.solutions

    noisy_tokens, noise_positions = apply_uniform_noise(
        targets, noise_ratio
    )
    editable_mask = build_editable_mask(noise_positions, editable_ratio)
    fix_mask = ~editable_mask

    logits = forward_in_batches(model, noisy_tokens, eval_batch_size)
    noise_loss, noise_acc = compute_masked_loss_and_accuracy(
        logits, targets, noise_positions
    )
    editable_loss, editable_acc = compute_masked_loss_and_accuracy(
        logits, targets, editable_mask
    )
    
    sequences = run_diffusion_sampling(
        model=model,
        input_ids=noisy_tokens,
        fix_mask=fix_mask,
        steps=step,
        algorithm=algorithm,
        remask_scheduler=remask_scheduler,
        batch_size=eval_batch_size,
        confidence_threshold=confidence_threshold,
        mad_k=mad_k,
    )
    preds = sequences.long().to(device)
    board_acc = compute_board_accuracy(preds, targets).item()

    metrics_dict = {
        "noise_ratio": float(noise_ratio),
        "editable_ratio": float(editable_ratio),
        "step": int(step),
        "algorithm": algorithm,
        "noise_token_loss": noise_loss.item(),
        "noise_token_accuracy": noise_acc.item(),
        "editable_token_loss": editable_loss.item(),
        "editable_token_accuracy": editable_acc.item(),
        "noise_tokens_per_sample": (
            noise_positions.sum(dim=1).float().mean().item()
        ),
        "editable_tokens_per_sample": (
            editable_mask.sum(dim=1).float().mean().item()
        ),
        "board_accuracy": board_acc,
    }
    if confidence_threshold is not None:
        metrics_dict["confidence_threshold"] = float(confidence_threshold)
    if mad_k is not None:
        metrics_dict["mad_k"] = float(mad_k)

    log_metrics(metrics_dict)

    return metrics_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniform noise diffusion evaluation",
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
        "--noise-ratio",
        type=float,
        required=True,
        help="Noise ratio",
    )
    parser.add_argument(
        "--editable-ratio",
        type=float,
        required=True,
        help="Editable ratio",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Diffusion steps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Sampling algorithm",
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
        "--remask-scheduler",
        type=str,
        default="linear",
        help="Remask scheduler",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2000,
        help="Batch size for model/diffusion evaluation",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resources = load_model_and_data(
        args.checkpoint,
        args.data_dir,
        device,
        args.num_samples,
    )

    config = {
        "mode": "uniform_noise_diffusion",
        "noise_ratio": args.noise_ratio,
        "editable_ratio": args.editable_ratio,
        "step": args.step,
        "algorithm": args.algorithm,
        "remask_scheduler": args.remask_scheduler,
        **resources.checkpoint_meta,
    }

    wandb.init(project=args.project, name=args.run_name, config=config)
    evaluate_uniform_noise_with_diffusion(
        resources=resources,
        noise_ratio=args.noise_ratio,
        editable_ratio=args.editable_ratio,
        step=args.step,
        algorithm=args.algorithm,
        remask_scheduler=args.remask_scheduler,
        eval_batch_size=args.eval_batch_size,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
