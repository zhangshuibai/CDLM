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

## Paper configuration

Appendix F does not state the following settings. The paper runs, and the scripts
in `scripts/`, use:

- replacement probability `mixture_prob` = 0.1 (α in the paper's Eq. (1));
- loss weight 1.0 on replaced cells (λ_noise in Eq. (1));
- a mean over replaced cells: the loss is the mean cross-entropy over masked cells
  plus 1.0 × the mean cross-entropy over replaced cells, each mean taken over all
  such cells in the batch. This is not the unnormalised sum over replaced cells
  written in the paper's Eq. (1);
- checkpoint selection on the test boards: `best_model.pt` is chosen on the same
  2,000 test boards that the evaluation reports on (see [2. Train](#2-train)).

## Model

`model.py` defines the model that the paper and code call a DiT. It is a
bidirectional pre-norm Transformer: 12 layers, hidden size
512, 8 heads, MLP ratio 4, dropout 0.1, learned absolute position embeddings, and
no timestep conditioning. A board is a sequence of 81 tokens over a vocabulary of
10: `0` is `[MASK]` and `1`-`9` are digits.

The model has **37,881,344 parameters** (37.9M), counted on CPU from this
configuration: 12 blocks × 3,152,384 = 37,828,608, plus 41,472 position-embedding,
5,120 token-embedding, 1,024 final-norm and 5,120 output-head parameters. The
"28.6M" in the `model.py` docstrings and in a `train.py` comment is stale; the count
that `create_sudoku_dit()` prints at runtime is the correct one.

## Setup

Python 3.10 or newer is required (`eval/` uses `int | None` annotations). All
commands in this README are run from the repository root:

```bash
pip install -r sudoku/requirements.txt
```

(Equivalently, `cd sudoku && pip install -r requirements.txt`.)

The scripts can also be launched from any other directory; they locate `sudoku/`
on their own. Make sure `python` on your `PATH` is the environment you installed
into, since the scripts and the sweep launch `python` subprocesses. `eval/` imports
the repository's top-level `llada_sample.py`, the same sampler used for the code
experiments, so keep this directory inside the repository.

Quick CPU check (a few seconds, random weights):

```bash
python sudoku/smoke_test.py
```

### Environment variables

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `DATA_DIR` | `sudoku/data` | `generate_data.sh`, `train.sh`, `eval_*.sh` | generated boards |
| `CKPT_ROOT` | `sudoku/checkpoints` | `train.sh`, `eval_*.sh` | checkpoints (see the note on disk use below) |
| `RESULTS_ROOT` | `sudoku/results` | `eval_*.sh`, `make_figures.sh` | sweep results and figures |
| `MIXTURE_PROB` | `0.1` | `train.sh`, `eval_*.sh`, `make_figures.sh` | replacement probability of the CDLM run; write it as Python prints it (`0.1`, `0.02`, `0.2`), since it names the `prob_<value>` directory |
| `CHECKPOINT` | `best_model.pt` | `eval_*.sh`, `make_figures.sh` | checkpoint file that is evaluated and plotted |
| `DEVICE` | `cuda` | `eval_*.sh` | evaluation device. Without a usable GPU, `run_all_evals.py` falls back to the CPU without a warning. Training does not read it: `train.py` uses the first visible GPU, or the CPU if there is none |
| `NUM_WORKERS` | `8` | `train.sh` | DataLoader workers for the train and test loaders (the paper runs used 64; this does not change the sampled data) |
| `TRAIN_SIZE`, `TEST_SIZE` | `36000`, `2000` | `generate_data.sh` | number of training and test boards |
| `GEN_WORKERS` | all cores | `generate_data.sh` | generator processes |
| `MODES` | `uniform_noise_only uniform_noise_diffusion` for `eval_refinement.sh`, `absorbing` for `eval_completion.sh` | `eval_*.sh` | space-separated subset of `absorbing`, `uniform_noise_only`, `uniform_noise_diffusion` to run with that script's grid |
| `FIG_DIR` | `$RESULTS_ROOT/figures` | `make_figures.sh` | output directory of the figures |
| `WANDB_MODE` | `disabled` | `train.sh`, `eval_*.sh` | set to `online` or `offline` to log to Weights & Biases |
| `WANDB_PROJECT`, `WANDB_PROJECT_EVAL` | `sudoku-diffusion-lm`, `sudoku_eval` | `train.sh`, `eval_*.sh` | W&B project names |

Relative paths are resolved against the directory you launch from. Choose the GPU
with `CUDA_VISIBLE_DEVICES`.

Script arguments: `train.sh absorbing|mixture [args]` and `generate_data.sh [args]`
append any extra arguments to the `train.py` or `generate_data.py` command line,
where they override the script's values (for example `--num-steps 2000`, or
`--force` to regenerate data). The other scripts take no arguments.

Only the scripts in `scripts/` use the paper configuration. The defaults of the
Python entry points are stale: `generate_data.py` defaults to 48,000 training
boards (its docstring says 48k), and `train.py` to batch size 128, evaluation batch
size 128, 100,000 steps, 4 workers and `--loss-mode absorbing`. The paper uses
36,000 training boards, batch size 64, evaluation batch size 2,000 and 200,000
steps, as set by the scripts.

## 1. Generate data

```bash
bash sudoku/scripts/generate_data.sh
```

This writes 36,000 training and 2,000 test boards to `DATA_DIR`. Training and
evaluation read only `*_solutions.npy` (complete, valid grids); the puzzle files are
a by-product of the generator. Generation is not seeded, so the boards differ from
the ones used for the paper, which are not included in this repository. A split
that already exists in `DATA_DIR` with at least the requested number of boards is
kept; pass `--force` to regenerate it. Generation took about 0.5 CPU-seconds per
board in our check, so about 5 CPU-hours for the full split, spread over
`GEN_WORKERS` processes.

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
  The loss is `CE(masked cells) + 1.0 * CE(replaced cells)`, where each CE is a
  mean over the batch's cells of that kind (see [Paper configuration](#paper-configuration)).
  Visible clean cells receive no loss.

Once before training and then every 1,000 steps, the model is scored on the test
boards at a fixed masking ratio of 0.5. The score is token accuracy on the masked
cells; for CDLM it also includes the replaced cells. `best_model.pt` is written
only when this score beats the best score so far, starting from the score of the
untrained model. It is the checkpoint used in the paper. Note that this selection
uses the same test boards that the evaluation later reports on. Two consequences:

- A run shorter than 1,000 steps (`--eval-interval`) writes no `best_model.pt`. For
  short runs, evaluate `final_model.pt` instead (`CHECKPOINT=final_model.pt`).
- `train.py` does not clear its output directory. A `best_model.pt` left by an
  earlier run stays there unless the new run improves on its own initial score.

Checkpoints are also written every 10,000 steps, plus `final_model.pt` at the end.
Each checkpoint includes the optimiser state (about 0.45 GB), so one run takes
about 10 GB. Point `CKPT_ROOT` at a larger disk if needed; a symlink works. The
training time of the paper runs was not recorded.

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
2,000 test boards per point (fewer if `TEST_SIZE` is smaller). Each point is a
separate `eval/run_all_evals.py` process that reloads the checkpoint and the test
boards. Results are written to
`$RESULTS_ROOT/{refinement,completion}/<run dir>/<checkpoint stem>/<mode>_results.{csv,json}`,
where `<run dir>` is `checkpoints_absorbing` or
`uniform_absorbing_mixture_checkpoints/prob_<p>` and `<checkpoint stem>` is, for
example, `best_model`. The sweep's output is saved next to them as
`refinement_sweep.log` or `completion_sweep.log`.

| Sweep | Mode | Grid | Processes per checkpoint |
| --- | --- | --- | --- |
| refinement | `uniform_noise_only` | noise ratio 0.3, 0.2, 0.1 | 3 |
| refinement | `uniform_noise_diffusion` | noise {0.1, 0.2, 0.3} × editable {0.4, 0.5, 0.6}; `llada_sample` steps T = 1-4; random-remask, τ ∈ {-1, 0.95, 0.9, 0.8}, MAD k ∈ {2.0, 2.5, 3.0} | 288 |
| completion | `absorbing` | mask ratio {0.3, 0.4, 0.5, 0.6}; T = 1-16; random-remask, τ ∈ {-1, 0.95, 0.9, 0.8, 0.7} | 384 |

That is 675 processes per checkpoint and about 1,350 for the two checkpoints.

`sweep_eval.py` is unchanged from the original code. Its built-in grid is the
completion grid. The refinement results came from an earlier revision that
differed only in the step list and the algorithm list. `eval/sweep_presets.py`
restores those two lists (`--preset refinement`) and changes nothing else. With
`MODES=absorbing`, `eval_refinement.sh` therefore runs the absorbing mode with
steps 1-4 and the refinement algorithm list, as the original refinement run did;
no figure uses that output.

### Runtime

The timestamps of the paper's result files give the wall-clock times below. The
sweeps ran on a machine with NVIDIA A100-PCIE-40GB GPUs, several at a time, which
explains the spread.

| Sweep | Wall-clock per checkpoint |
| --- | --- |
| refinement, `uniform_noise_only` (3 points) | about 25 s |
| refinement, `uniform_noise_diffusion` (288 points) | 1.2 to 3 h |
| completion, `absorbing` (384 points) | 1.75 to 3.75 h |

A `uniform_noise_only` point, which needs a single forward pass, took about 8 s, so
starting a process and loading the checkpoint costs at most that much per point.
Running both scripts for both checkpoints one after another takes roughly 6 to 14
hours. The two
scripts write to different directories, so they can run at the same time on two
GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 bash sudoku/scripts/eval_refinement.sh &
CUDA_VISIBLE_DEVICES=1 bash sudoku/scripts/eval_completion.sh &
wait
```

### Failure handling

`sweep_eval.py` is byte-identical to the paper code, and two of its behaviours are
kept:

- It exits with status 0 even when evaluation points fail. It prints a warning
  with the failing point's stderr, records the point without metrics, and ends with
  `SWEEP COMPLETE: N total runs, M failed`.
- If a point's `EVAL_METRICS_JSON=` line cannot be parsed, it calls
  `pdb.set_trace()`, a leftover debugging statement, and waits for debugger input.

The scripts handle both. `run_sweep` in `scripts/common.sh` starts the sweep with
standard input from `/dev/null`, so the debugger quits at once and the sweep aborts
with an error instead of waiting. After each checkpoint's sweep, it requires the
summary line to report 0 failed runs, and every row of each requested mode's CSV
to have its metric (`board_accuracy`, or `clean_token_confidence` for
`uniform_noise_only`). Otherwise the script stops with a non-zero exit status and
names the log to inspect. If you call `sweep_eval.py` or `sweep_presets.py`
directly, check its summary line yourself.

### Evaluation modes

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

`llada_sample.py` (temperature 0) runs T steps with threshold τ. For
t = 0, ..., T-1:

1. Run one forward pass on the current board z. Fill every `[MASK]` in E with the
   argmax token over all 10 tokens.
2. Set the confidence of each cell to c_i = p(z_i | z) for the token now in cell
   i, using the logits of this pass.
3. The remask set R is {i ∈ E : c_i ≤ τ}. With τ = -1 ("neg10" in figure
   filenames), a linear schedule is used instead: R is the ⌊|E|·(T−t−1)/T⌋
   least-confident cells of E, and `random-remask` applies the same schedule with
   random scores. MAD (`self_conf-remask:vanilla_MAD`) takes as R the cells of E
   whose confidence is more than k × 1.4826 × MAD below the median confidence over
   E. With a threshold or MAD, a board whose R is empty is frozen for the
   remaining steps.
4. Unless t = T−1, reset the cells in R to `[MASK]`.

Visible editable cells that are not remasked keep their digit, so every change
passes through `[MASK]`. Cells outside E never change. For uniform noise, T = 1
returns the corrupted board unchanged, and T = k+1 performs k remask-and-repredict
rounds. For completion, T = 1 fills all masks in one pass.

The paper's Algorithm 2 describes this procedure, but its pseudo-code differs from
the code in these ways:

| | Algorithm 2 (paper) | Code |
| --- | --- | --- |
| confidence | c_i = max over digits of p(v \| z) | c_i = p(z_i \| z) of the token currently in cell i |
| prediction | argmax over the nine digits | argmax over all 10 tokens, including `[MASK]`; a cell whose argmax is `[MASK]` stays masked |
| editable cells not in R | overwritten with the argmax prediction | keep their current digit |
| remask test | c_i < τ | c_i ≤ τ |
| last step | remasks at t = T−1 as well, so the output can contain `[MASK]` | never remasks at t = T−1 |
| empty R | no special case | the board is frozen for the remaining steps (with a deterministic forward pass this changes nothing unless the board still holds a `[MASK]`) |
| steps shown in the figures | T | the uniform-noise figure's "Refined Steps" axis shows T−1 (0-3 for T = 1-4); the completion figure's "Sampling Steps" axis shows T |

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
bash sudoku/scripts/make_figures.sh      # -> $FIG_DIR (default $RESULTS_ROOT/figures/)
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

## Quick run

To check the pipeline end to end on a GPU before the full run (this does not
reproduce the paper's numbers):

```bash
export DATA_DIR=sudoku/quick/data CKPT_ROOT=sudoku/quick/checkpoints RESULTS_ROOT=sudoku/quick/results
TRAIN_SIZE=2000 TEST_SIZE=200 bash sudoku/scripts/generate_data.sh
bash sudoku/scripts/train.sh absorbing --num-steps 2000
bash sudoku/scripts/train.sh mixture --num-steps 2000
CHECKPOINT=final_model.pt MODES=uniform_noise_only bash sudoku/scripts/eval_refinement.sh
```

The last command runs 3 points per checkpoint. The completion sweep and the
`uniform_noise_diffusion` mode have no smaller preset, and `make_figures.sh` needs
all three modes.

## Reproducibility notes

- The evaluation code sets no seed, so every sweep point draws fresh corruption.
  Training is seeded, but CUDA kernels are not deterministic. Expect small
  differences from the paper, which reports board accuracy on 2,000 boards.
- `best_model.pt` is selected on the test boards, as in the paper runs.
- The figures use Times New Roman when it is installed, and DejaVu Serif otherwise.

## Files

| Path | Role |
| --- | --- |
| `generate_data.py`, `advanced_sudoku_generator.py`, `puzzle_generator.py` | board generation (the generator classes are third-party, MIT; see `THIRD_PARTY_NOTICES.md`) |
| `model.py` | Transformer denoiser |
| `train.py` | training for both noise settings |
| `eval/utils.py` | corruption, editable-set construction, sampler wrapper |
| `eval/absorbing_eval.py`, `eval/uniform_noise_eval.py`, `eval/uniform_noise_diffusion_eval.py` | the three evaluation modes |
| `eval/run_all_evals.py` | one evaluation point |
| `eval/sweep_eval.py`, `eval/sweep_presets.py` | sweeps over the grids above |
| `eval/analysis/compare_*.py` | paper figures |
| `scripts/*.sh` | entry points described above |
| `smoke_test.py` | CPU sanity check |

All Python files except `eval/sweep_presets.py` and `smoke_test.py` are
byte-identical to the code used for the paper.
