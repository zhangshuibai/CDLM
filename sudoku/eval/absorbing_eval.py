#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation under absorbing-mask noise with diffusion sampling.
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
    apply_absorbing_noise,
    compute_board_accuracy,
    compute_masked_loss_and_accuracy,
    forward_in_batches,
    load_model_and_data,
    log_metrics,
    run_diffusion_sampling,
)


def evaluate_absorbing(
    resources: EvaluationResources,
    mask_ratio: float,
    step: int,
    algorithm: str,
    remask_scheduler: str,
    eval_batch_size: int | None,
    confidence_threshold: float | None = None,
    mad_k: float | None = None,
) -> dict:
    model = resources.model
    device = resources.device
    targets = resources.solutions

    masked_inputs, mask_positions = apply_absorbing_noise(targets, mask_ratio)
    fix_mask = ~mask_positions

    logits = forward_in_batches(
        model, masked_inputs, eval_batch_size
    )
    loss, acc = compute_masked_loss_and_accuracy(
        logits, targets, mask_positions
    )
    masked_mean = mask_positions.sum(dim=1).float().mean().item()

    sequences = run_diffusion_sampling(
        model=model,
        input_ids=masked_inputs,
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
        "mask_ratio": float(mask_ratio),
        "step": int(step),
        "algorithm": algorithm,
        "token_loss": loss.item(),
        "token_accuracy": acc.item(),
        "masked_tokens_per_sample": masked_mean,
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
        description="Absorbing noise evaluation with diffusion sampling",
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
        "--mask-ratio",
        type=float,
        required=True,
        help="Mask ratio",
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
        "mode": "absorbing_diffusion",
        "mask_ratio": args.mask_ratio,
        "step": args.step,
        "algorithm": args.algorithm,
        "remask_scheduler": args.remask_scheduler,
        **resources.checkpoint_meta,
    }

    wandb.init(project=args.project, name=args.run_name, config=config)
    evaluate_absorbing(
        resources=resources,
        mask_ratio=args.mask_ratio,
        step=args.step,
        algorithm=args.algorithm,
        remask_scheduler=args.remask_scheduler,
        eval_batch_size=args.eval_batch_size,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
