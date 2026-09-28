#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation under uniform replacement noise without diffusion.
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
    compute_masked_loss_and_accuracy,
    forward_in_batches,
    load_model_and_data,
    log_metrics,
)


def evaluate_uniform_noise_only(
    resources: EvaluationResources,
    noise_ratio: float,
    eval_batch_size: int | None,
) -> dict:
    model = resources.model
    targets = resources.solutions

    noisy_tokens, noise_positions = apply_uniform_noise(targets, noise_ratio)
    logits = forward_in_batches(model, noisy_tokens, eval_batch_size)
    probs = torch.softmax(logits.float(), dim=-1)
    current_tokens = noisy_tokens
    token_probs = torch.gather(
        probs, dim=-1, index=current_tokens.unsqueeze(-1)
    ).squeeze(-1)

    noise_loss, noise_acc = compute_masked_loss_and_accuracy(
        logits, targets, noise_positions
    )
    noise_conf_mean = (
        token_probs[noise_positions].mean().item()
        if noise_positions.any()
        else 0.0
    )
    clean_positions = ~noise_positions
    clean_loss, clean_acc = compute_masked_loss_and_accuracy(
        logits, targets, clean_positions
    )
    clean_conf_mean = (
        token_probs[clean_positions].mean().item()
        if clean_positions.any()
        else 0.0
    )

    log_metrics(
        {
            "noise_ratio": float(noise_ratio),
            "noise_token_loss": noise_loss.item(),
            "noise_token_accuracy": noise_acc.item(),
            "clean_token_loss": clean_loss.item(),
            "clean_token_accuracy": clean_acc.item(),
            "noise_tokens_per_sample": (
                noise_positions.sum(dim=1).float().mean().item()
            ),
            "noise_token_confidence": noise_conf_mean,
            "clean_token_confidence": clean_conf_mean,
        }
    )

    return {
        "noise_ratio": float(noise_ratio),
        "noise_token_loss": float(noise_loss.item()),
        "noise_token_accuracy": float(noise_acc.item()),
        "clean_token_loss": float(clean_loss.item()),
        "clean_token_accuracy": float(clean_acc.item()),
        "noise_tokens_per_sample": float(
            noise_positions.sum(dim=1).float().mean().item()
        ),
        "noise_token_confidence": float(noise_conf_mean),
        "clean_token_confidence": float(clean_conf_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniform noise evaluation without diffusion",
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
        "--eval-batch-size",
        type=int,
        default=2000,
        help="Batch size for model evaluation",
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
        "mode": "uniform_noise_only",
        "noise_ratio": args.noise_ratio,
        **resources.checkpoint_meta,
    }

    wandb.init(project=args.project, name=args.run_name, config=config)
    evaluate_uniform_noise_only(
        resources=resources,
        noise_ratio=args.noise_ratio,
        eval_batch_size=args.eval_batch_size,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
