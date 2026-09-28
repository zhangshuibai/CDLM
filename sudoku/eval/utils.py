#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions shared across Sudoku evaluation scripts.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb

SUDOKU_ROOT = Path(__file__).resolve().parents[1]
DLMA_ROOT = Path(__file__).resolve().parents[2]

for path in (SUDOKU_ROOT, DLMA_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from model import create_sudoku_dit  # noqa: E402
from llada_sample import llada_sample  # noqa: E402

MASK_ID = 0
SEQ_LENGTH = 81
VOCAB_SIZE = 10


@dataclass
class EvaluationResources:
    model: torch.nn.Module
    solutions: torch.Tensor  # shape: (B, SEQ_LENGTH)
    device: torch.device
    checkpoint_meta: Dict


def load_model_and_data(
    checkpoint_path: str,
    data_dir: str,
    device: torch.device,
    num_samples: int,
) -> EvaluationResources:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_sudoku_dit(vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    solutions_np = np.load(Path(data_dir) / "test_solutions.npy")[:num_samples]
    solutions = torch.from_numpy(solutions_np).long().to(device)

    checkpoint_meta = {
        "step": checkpoint.get("step", None),
        "best_test_acc": checkpoint.get("best_test_acc", None),
        "config": checkpoint.get("config", {}),
        "checkpoint_path": str(checkpoint_path),
    }

    return EvaluationResources(
        model=model,
        solutions=solutions,
        device=device,
        checkpoint_meta=checkpoint_meta,
    )


def ensure_at_least_one(mask: torch.Tensor) -> torch.Tensor:
    """
    Guarantee each sample has at least one True entry.
    mask: (B, N) boolean tensor
    """
    mask_sums = mask.sum(dim=1) == 0
    batch_indices = torch.nonzero(mask_sums, as_tuple=False).flatten()
    if batch_indices.numel() > 0:
        num_positions = mask.shape[1]
        random_cols = torch.randint(
            0,
            num_positions,
            (batch_indices.numel(),),
            device=mask.device,
        )
        mask[batch_indices, random_cols] = True
    return mask


def apply_absorbing_noise(
    targets: torch.Tensor,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask tokens with the given ratio.
    Returns masked_tokens, mask_positions
    """
    rand = torch.rand_like(targets, dtype=torch.float32)
    mask_positions = rand < ratio
    mask_positions = ensure_at_least_one(mask_positions.clone())

    masked_tokens = targets.clone()
    masked_tokens[mask_positions] = MASK_ID
    return masked_tokens, mask_positions


def apply_uniform_noise(
    targets: torch.Tensor,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace tokens uniformly (excluding MASK_ID and the original token).
    Returns noisy_tokens, noise_positions
    """
    rand = torch.rand_like(targets, dtype=torch.float32)
    noise_positions = rand < ratio
    noise_positions = ensure_at_least_one(noise_positions.clone())

    noisy_tokens = targets.clone()
    originals = targets[noise_positions]

    if originals.numel() > 0:
        new_tokens = torch.randint(
            0,
            VOCAB_SIZE - 1,
            size=(originals.numel(),),
            device=targets.device,
        )
        new_tokens = new_tokens + (new_tokens >= MASK_ID).long()

        clashes = new_tokens == originals
        while clashes.any():
            fresh = torch.randint(
                0,
                VOCAB_SIZE - 1,
                size=(clashes.sum().item(),),
                device=targets.device,
            )
            fresh = fresh + (fresh >= MASK_ID).long()
            new_tokens[clashes] = fresh
            clashes = new_tokens == originals

        noisy_tokens[noise_positions] = new_tokens
    else:
        raise ValueError(f"Invalid ratio: {ratio}")

    return noisy_tokens, noise_positions


def build_editable_mask(
    noise_mask: torch.Tensor,
    target_ratio: float,
) -> torch.Tensor:
    """
    Construct editable positions ensuring all noise tokens are included.
    target_ratio is a guideline for the desired fraction of editable positions.
    """
    batch_size, seq_len = noise_mask.shape
    editable = noise_mask.clone()
    noise_counts = noise_mask.sum(dim=1).long()

    # Build per-sample desired editable counts tensor
    target_tensor = torch.full(
        (batch_size,),
        float(target_ratio),
        device=noise_mask.device,
        dtype=torch.float32,
    )
    desired_counts = torch.clamp(
        (target_tensor * seq_len).round().long(),
        min=0,
        max=seq_len,
    )

    for idx in range(batch_size):
        current_count = editable[idx].sum().item()
        desired = max(noise_counts[idx].item(), desired_counts[idx].item())
        desired = min(desired, seq_len)

        if current_count >= desired:
            continue

        available = torch.nonzero(~editable[idx], as_tuple=False).flatten()
        if available.numel() == 0:
            continue

        need = min(desired - current_count, available.numel())
        perm = torch.randperm(
            available.numel(),
            device=noise_mask.device,
        )
        chosen = available[perm[:need]]
        editable[idx, chosen] = True

    return editable


def compute_masked_loss_and_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross-entropy loss and accuracy on the masked subset.
    """
    flat_mask = mask.view(-1)
    total = flat_mask.sum()
    if total.item() == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero

    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    selected_logits = flat_logits[flat_mask]
    selected_targets = flat_targets[flat_mask]

    loss = F.cross_entropy(selected_logits, selected_targets)
    preds = selected_logits.argmax(dim=-1)
    acc = (preds == selected_targets).float().mean()
    return loss, acc


def compute_board_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (preds == targets).all(dim=1).float().mean()


def run_diffusion_sampling(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    fix_mask: torch.Tensor,
    steps: int,
    algorithm: str,
    remask_scheduler: str = "linear",
    batch_size: int | None = None,
    confidence_threshold: float | None = None,
    mad_k: float | None = None,
) -> torch.Tensor:
    """
    Run llada_sample and return final sequences.
    """
    with torch.no_grad():
        result = llada_sample(
            model=model,
            input_ids=input_ids.clone(),
            fix_mask=fix_mask,
            mask_id=MASK_ID,
            attention_mask=None,
            steps=steps,
            algorithm=algorithm,
            temperature=0.0,
            top_p=None,
            top_k=None,
            remask_scheduler=remask_scheduler,
            confidence_threshold=confidence_threshold,
            mad_k=mad_k,
            return_history=False,
        )
    return result["sequences"]


def init_wandb_run(
    project: str,
    run_name: str,
    config: Dict,
) -> wandb.sdk.wandb_run.Run:
    return wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
    )


def log_metrics(data: Dict) -> None:
    wandb.log(data)


def mask_ratio_grid() -> Tuple[Iterable[float], Iterable[float]]:
    absorbing_ratios = [round(x, 1) for x in np.linspace(0.2, 0.9, 8)]
    head = [0.5, 0.4, 0.3, 0.2, 0.1]
    tail = [round(x, 2) for x in np.linspace(0.09, 0.01, 9)]
    uniform_ratios = head + tail
    uniform_ratios = sorted(uniform_ratios, reverse=True)
    return absorbing_ratios, uniform_ratios


def forward_in_batches(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        return model(tokens).logits
