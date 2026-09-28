#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU sanity check for the Sudoku experiment: builds the DiT, runs one forward and
backward pass under both training noises, and one editable-region refinement
call through llada_sample. Weights are random, so accuracies are meaningless.
"""

import numpy as np
import torch

from eval.utils import (
    MASK_ID,
    apply_absorbing_noise,
    apply_uniform_noise,
    build_editable_mask,
    run_diffusion_sampling,
)
from model import create_sudoku_dit
from puzzle_generator import PuzzleGenerator
from train import (
    add_absorbing_noise,
    add_uniform_absorbing_mixture_noise,
    compute_absorbing_loss,
)

EXPECTED_PARAMS = 37_881_344


def solved_boards(n):
    gen = PuzzleGenerator()
    boards = []
    for _ in range(n):
        grid = np.zeros((9, 9), dtype=int)
        gen.fill_grid(grid)
        boards.append(grid.flatten())
    return torch.from_numpy(np.stack(boards)).long()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model = create_sudoku_dit(vocab_size=10, seq_length=81)
    n_params = model.count_parameters()["total"]
    assert n_params == EXPECTED_PARAMS, n_params

    boards = solved_boards(8)
    assert boards.min() >= 1 and boards.max() <= 9

    logits = model(boards).logits
    assert logits.shape == (8, 81, 10)

    # Training noises and losses (one backward pass, no optimizer step).
    model.train()
    x, mask = add_absorbing_noise(boards, MASK_ID, 0.2, 0.9)
    assert (x[mask] == MASK_ID).all() and (x[~mask] == boards[~mask]).all()
    loss, _ = compute_absorbing_loss(model(x).logits, boards, mask)
    loss.backward()

    model.zero_grad()
    x, mask, mix = add_uniform_absorbing_mixture_noise(
        boards, MASK_ID, 0.2, 0.9, mixture_prob=0.1, vocab_size=10
    )
    assert not (mask & mix).any()
    assert (x[mix] != boards[mix]).all() and (x[mix] != MASK_ID).all()
    out = model(x).logits
    mask_loss, _ = compute_absorbing_loss(out, boards, mask)
    mix_loss, _ = compute_absorbing_loss(out, boards, mix)
    (mask_loss + 1.0 * mix_loss).backward()

    # Editable-region refinement (uniform_noise_diffusion) and completion (absorbing).
    model.eval()
    noisy, noise = apply_uniform_noise(boards, 0.2)
    editable = build_editable_mask(noise, 0.5)
    assert (editable | ~noise).all()
    out = run_diffusion_sampling(
        model, noisy, fix_mask=~editable, steps=3,
        algorithm="self_conf-remask:vanilla", confidence_threshold=0.8,
    )
    assert (out[~editable] == noisy[~editable]).all()

    masked, mask = apply_absorbing_noise(boards, 0.5)
    out = run_diffusion_sampling(
        model, masked, fix_mask=~mask, steps=4,
        algorithm="self_conf-remask:vanilla", confidence_threshold=0.7,
    )
    assert (out[~mask] == boards[~mask]).all()

    print(f"OK: {n_params:,} parameters; forward, backward and refinement ran on CPU.")


if __name__ == "__main__":
    main()
