# Sudoku: a from-scratch controlled comparison

This directory reproduces the Sudoku experiment in Appendix F of the paper.

It is the only genuinely from-scratch controlled comparison in the paper. The
language-model backbones used elsewhere are either initialised from
autoregressive models or too large to retrain, so their corrective behaviour could
in principle be inherited from pretraining. Here, the same small Transformer is
trained twice from random initialisation, with identical data, optimiser, schedule,
seed and number of steps. The only difference is the noise process:

| Name in figures | `--loss-mode` | Training noise |
| --- | --- | --- |
| MDLM | `absorbing` | absorbing (mask-only) noise |
| CDLM | `uniform_absorbing_mixture` | absorbing noise plus uniform replacement of visible cells, `mixture_prob` = 0.1 |

## Model

`model.py` defines the model that the paper and code call a DiT. It is a
bidirectional pre-norm Transformer: 12 layers, hidden size
512, 8 heads, MLP ratio 4, dropout 0.1, learned absolute position embeddings, and
no timestep conditioning. A board is a sequence of 81 tokens over a vocabulary of
10: `0` is `[MASK]` and `1`-`9` are digits.

The model has **37,881,344 parameters** (37.9M), counted on CPU from this
configuration: 12 blocks × 3,152,384 = 37,828,608, plus 41,472 position-embedding,
5,120 token-embedding, 1,024 final-norm and 5,120 output-head parameters. The
"28.6M" in the `model.py` docstrings is stale; the count that `create_sudoku_dit()`
prints at runtime is the correct one.

## Setup

Python 3.10 or newer is required (`eval/` uses `int | None` annotations).

```bash
pip install -r requirements.txt
```

Run the scripts from any directory; they locate `sudoku/` themselves. Make sure
`python` on your `PATH` is the environment you installed into, since the sweep
launches `python` subprocesses. `eval/` imports the repository's top-level
`llada_sample.py`, the same sampler used for the code experiments, so keep this
directory inside the repository.

Quick CPU check (a few seconds, random weights):

```bash
python sudoku/smoke_test.py
```

All scripts read these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `sudoku/data` | generated boards |
| `CKPT_ROOT` | `sudoku/checkpoints` | checkpoints (see note on disk use below) |
| `RESULTS_ROOT` | `sudoku/results` | sweep results and figures |
| `MIXTURE_PROB` | `0.1` | replacement probability of the CDLM run |
| `CHECKPOINT` | `best_model.pt` | checkpoint that is evaluated and plotted |
| `DEVICE` | `cuda` | evaluation device |
| `NUM_WORKERS` | `8` | training DataLoader workers |
| `WANDB_MODE` | `disabled` | set to `online` or `offline` to log to Weights & Biases |
| `WANDB_PROJECT`, `WANDB_PROJECT_EVAL` | `sudoku-diffusion-lm`, `sudoku_eval` | W&B project names |

Choose the GPU with `CUDA_VISIBLE_DEVICES`.

## 1. Generate data

```bash
bash sudoku/scripts/generate_data.sh
```

This writes 36,000 training and 2,000 test boards to `DATA_DIR`. Training and
evaluation read only `*_solutions.npy` (complete, valid grids); the puzzle files are
a by-product of the generator. Generation is not seeded, so the boards differ from
the ones used for the paper, which are not included in this repository.

## 2. Train

```bash
bash sudoku/scripts/train.sh absorbing   # MDLM -> $CKPT_ROOT/checkpoints_absorbing/
bash sudoku/scripts/train.sh mixture     # CDLM -> $CKPT_ROOT/uniform_absorbing_mixture_checkpoints/prob_0.1/
```

Keep these directory names. The plotting scripts label a run "MDLM" when its path
contains `checkpoints_absorbing` and "CDLM" when it contains `prob_`.

Hyperparameters (identical for both runs):

| | |
| --- | --- |
| training boards / steps / batch size | 36,000 / 200,000 / 64 |
| optimiser | AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0.01, grad-norm clip 1.0 |
| schedule | cosine from 1e-4 to 1e-5 over 200,000 steps, no warmup |
| masking ratio | sampled per board, uniform in [0.2, 0.9] |
| seed | 42 |

How the two noises and losses work (`train.py`):

- **Absorbing.** Each board draws a ratio r ~ U[0.2, 0.9], and each cell is
  replaced by `[MASK]` independently with probability r. The loss is the mean
  cross-entropy over masked cells.
- **Mixture.** After the same masking, each still-visible cell is replaced
  independently with probability `mixture_prob` by a uniformly chosen wrong digit.
  The loss is `CE(masked cells) + 1.0 * CE(replaced cells)`. Visible clean cells
  receive no loss.

Every 1,000 steps the model is scored on the 2,000 test boards at a fixed masking
ratio of 0.5. For CDLM the score also includes the replaced cells. The best score
is saved as `best_model.pt`, which is the checkpoint used in the paper. Note that
this selection uses the same test boards that the evaluation later reports on.
Checkpoints are also written every 10,000 steps, plus `final_model.pt`. Each
checkpoint includes the optimiser state (about 0.45 GB), so one run takes about
10 GB. Point `CKPT_ROOT` at a larger disk if needed; a symlink works.

`train.py` also has a `uniform_absorbing_mixture_with_clean` mode, which adds a loss
on clean visible cells. The Sudoku appendix does not use it. Mixture probabilities
0.02 and 0.2 were also trained during development but are not reported. They can
be reproduced with `MIXTURE_PROB=0.02` or `MIXTURE_PROB=0.2`.

## 3. Evaluate

```bash
bash sudoku/scripts/eval_refinement.sh   # uniform-noise localisation and refinement
bash sudoku/scripts/eval_completion.sh   # absorbing-mask completion
```

Both scripts run `eval/sweep_eval.py` on the MDLM and CDLM checkpoints, using
2,000 test boards per point. Each point is a separate `eval/run_all_evals.py`
process. Results are written to
`$RESULTS_ROOT/{refinement,completion}/<checkpoint path>/<mode>_results.{csv,json}`.

| Sweep | Mode | Grid |
| --- | --- | --- |
| refinement | `uniform_noise_only` | noise ratio 0.3, 0.2, 0.1 |
| refinement | `uniform_noise_diffusion` | noise {0.1, 0.2, 0.3} × editable {0.4, 0.5, 0.6}; `llada_sample` steps T = 1-4; random-remask, τ ∈ {-1, 0.95, 0.9, 0.8}, MAD k ∈ {2.0, 2.5, 3.0} |
| completion | `absorbing` | mask ratio {0.3, 0.4, 0.5, 0.6}; T = 1-16; random-remask, τ ∈ {-1, 0.95, 0.9, 0.8, 0.7} |

`sweep_eval.py` is unchanged from the original code. Its built-in grid is the
completion grid. The refinement results came from an earlier revision that
differed only in the step list and the algorithm list. `eval/sweep_presets.py`
restores those two lists (`--preset refinement`) and changes nothing else.

The evaluation modes (`eval/utils.py`) are:

- **Uniform corruption.** Each cell is replaced independently with probability
  equal to the noise ratio (at least one cell per board) by a uniformly chosen
  wrong digit.
- **`uniform_noise_only`.** A single forward pass on the whole corrupted board.
  It reports the mean probability p(z_i | z) of the digit currently in the cell,
  separately over corrupted and clean cells. No editable set is involved.
- **`uniform_noise_diffusion`.** The editable set E is every corrupted cell plus
  randomly chosen clean cells, up to round(editable ratio × 81) cells. All other
  cells are fixed. The board is then refined with `llada_sample`.
- **`absorbing`.** Each cell is masked independently with probability equal to
  the mask ratio. The masked cells are the editable set.
- **Metric.** Board accuracy is the fraction of boards whose 81 cells all match
  the solution exactly.

### Refinement procedure as implemented

This is Algorithm 2 of the paper as run by `llada_sample.py` (temperature 0) with
T steps and threshold τ. For t = 0, ..., T-1:

1. Run one forward pass on the current board z. Fill every `[MASK]` in E with the
   argmax token.
2. Set the confidence of each cell to c_i = p(z_i | z) for the token now in cell
   i, using the logits of this pass.
3. The remask set R is {i ∈ E : c_i ≤ τ}. A board whose R is empty is frozen for
   the remaining steps. With τ = -1 ("neg10" in figure filenames), a linear
   schedule is used instead: R is the ⌊|E|·(T−t−1)/T⌋ least-confident cells of E.
   `random-remask` applies the same schedule with random scores.
4. Unless t = T−1, reset the cells in R to `[MASK]`.

Visible editable cells that are not remasked keep their digit, so every change
passes through `[MASK]`. Cells outside E never change. For uniform noise, T = 1
returns the corrupted board unchanged, and T = k+1 performs k remask-and-repredict
rounds. This is why the refinement figure's "Refined Steps" axis shows T−1
(0-3 for T = 1-4). For completion, T = 1 fills all masks in one pass, and the axis
shows T.

### Scope: the editable set is an oracle

The editable set E is built from the ground truth. It always contains every
corrupted cell, and every cell outside E is guaranteed correct and held fixed. The
model never has to find errors outside E. The refinement experiment therefore
validates confidence separation inside a region known to contain all errors: the
corrupted digits must receive lower confidence than the clean ones, so they are
remasked and repredicted. It does not validate end-to-end error localisation over
the full board. The `uniform_noise_only` measurement involves no editable set: it
measures confidence separation over the whole board in a single forward pass.

## 4. Figures

```bash
bash sudoku/scripts/make_figures.sh      # -> $RESULTS_ROOT/figures/
```

| Paper (Appendix F) | Output file | Setting |
| --- | --- | --- |
| Uniform Noise Corruption: Error-Token Localization | `sudoku_uniform_noise_comparison.pdf` | `uniform_noise_only`, noise ratio 0.1 / 0.2 / 0.3, confidence on clean vs corrupted cells (log scale; clean/noise ratio annotated) |
| Iterative Correction under Uniform Noise | `uniform_noise_diffusion_compare/sudoku_uniform_noise_diffusion_self_conf_remask_vanilla_confidence_threshold_08_comparison.pdf` | noise {0.1, 0.2} × editable {0.4, 0.5, 0.6}, τ = 0.8, refined steps 0-3 |
| Pure Completion under Masked Denoising | `absorbing_completion_compare/sudoku_absorbing_self_conf_remask_vanilla_confidence_threshold_07_comparison.pdf` | mask ratio 0.3-0.6, τ = 0.7, sampling steps 1-8 |

The same run also writes figures for the other algorithm settings in each sweep
(other thresholds such as `_08`, `_09`, `_095` and `_neg10`, random-remask, and
MAD). The paper does not show these.

To run everything in sequence, including data generation and both training runs:

```bash
bash sudoku/scripts/run_all.sh
```

## Reproducibility notes

- The evaluation code sets no seed, so every sweep point draws fresh corruption.
  Training is seeded, but CUDA kernels are not deterministic. Expect small
  differences from the paper, which reports board accuracy on 2,000 boards.
- The figures use Times New Roman when it is installed, and DejaVu Serif otherwise.

## Files

| Path | Role |
| --- | --- |
| `generate_data.py`, `advanced_sudoku_generator.py`, `puzzle_generator.py` | board generation (the generator classes are third-party; see `THIRD_PARTY_NOTICES.md`) |
| `model.py` | Transformer denoiser |
| `train.py` | training for both noise settings |
| `eval/utils.py` | corruption, editable-set construction, sampler wrapper |
| `eval/absorbing_eval.py`, `eval/uniform_noise_eval.py`, `eval/uniform_noise_diffusion_eval.py` | the three evaluation modes |
| `eval/run_all_evals.py` | one evaluation point |
| `eval/sweep_eval.py`, `eval/sweep_presets.py` | sweeps over the grids above |
| `eval/analysis/compare_*.py` | paper figures |
| `scripts/*.sh` | entry points described above |
| `smoke_test.py` | CPU sanity check |
