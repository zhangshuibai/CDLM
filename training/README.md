# Corrective training

This directory contains the code used to train the corrective diffusion language models (CDLM) in the
paper and their absorbing-only baselines (MDLM):

- **Open-dCoder-0.5B** (continued training of `fredzzp/open-dcoder-0.5B`): the headline CDLM-0.5B and
  MDLM-0.5B checkpoints, the mixture-probability sweep, the clean-token ablation and the seed study.
- **LLaDA-8B-Base LoRA transfer**: [`llada8b_lora/`](llada8b_lora/README.md).
- **Sudoku** (from-scratch controlled comparison, Appendix F): [`../sudoku/`](../sudoku/README.md).

Not included in this release: the 16B **LLaDA2.0-mini** math fine-tuning (Nemotron-Math-v2) and the
**ParallelBench** fine-tuning.

Figure, table and appendix numbers below refer to the NeurIPS 2026 submission. The seed study and
the LLaDA-8B transfer are not in the submission: they are added in the camera-ready version
(NeurIPS 2026), and their numbers were also reported in the public OpenReview discussion of the
submission.

## Contents

| Path | What it is |
| --- | --- |
| `veomni/`, `tasks/train_torch.py` | Training framework: a snapshot of Open-dLLM / VeOmni with the mixture objective added. `train_torch.py` is the entry point of every 0.5B run. |
| `configs/pretrain/qwen2_5_coder_500M.yaml` | The only config used by the 0.5B runs. The launch scripts override parts of it on the command line. |
| `configs/cdlm/` | The fully resolved configurations of the runs behind CDLM-0.5B and MDLM-0.5B, for reference (see [Provenance](#provenance-of-cdlm-05b-and-mdlm-05b)). |
| `scripts/` | Launchers for the 0.5B experiments. |
| `tools/` | Conversion of checkpoints into HuggingFace directories for the CRB pipeline. |
| `data_prep/` | Download and verification of the training corpus. |
| `llada8b_lora/` | LLaDA-8B-Base LoRA trainer and its CRB evaluation scripts. |
| `requirements.txt` | Pinned environment for the 0.5B runs. |
| `LICENSE`, `NOTICE` | Apache-2.0 license and attribution for the code derived from VeOmni / Open-dLLM. |
| `PATCH_vs_Open-dLLM-5b9ba4d.diff` | The complete difference between this snapshot and the public Open-dLLM at commit 5b9ba4d (9 files). Applying it to that commit reproduces `veomni/` and `tasks/train_torch.py` exactly. The other upstream entry points (`tasks/infer.py`, `tasks/omni/`) are not shipped. |

## Objective as implemented

The objective is selected by three flags: `--train.mixture_prob` (α), `--train.noise_token_wt` and
`--train.clean_token_wt`. The paper settings are:

| Model | `mixture_prob` | `noise_token_wt` | `clean_token_wt` |
| --- | --- | --- | --- |
| MDLM (absorbing only) | 0 | 0 | 0 |
| CDLM | 0.1 | 0.1 | 0 |
| CDLM + ctw (Table 4) | 0.1 | 0.1 | 0.1 |
| α sweep (Fig. 6) | α | 0.1 | 0 |

`repr_align_wt` is 0 in every run, so the teacher-alignment code in the model is never used. The
embedding and output layers are frozen (`--train.freeze_layers=lm_head,embed_tokens`).

### Corruption

`process_pretrain_example` (`veomni/data/data_transform.py`) tokenizes each document's `text`, appends
EOS and cuts the result into chunks of at most 4096 tokens. Chunks are packed into micro-batches of
up to 3 × 4096 tokens. `DataCollatorWithPositionIDsMixture_Masking` (`veomni/data/data_collator.py`)
then corrupts each chunk independently:

1. **Masking.** Draw u ~ U(0, 1) and set the mask ratio t = clamp(u, 1/500, 1 − 1/500). Every token
   is replaced by the mask token (`<M>`, id 151665) independently with probability t.
2. **Uniform replacement** (only when `mixture_prob` > 0). Every token that was not masked is
   selected independently with probability `mixture_prob`. A selected token is replaced by a token
   drawn uniformly from ids 0 to `vocab_size` − 2, where `vocab_size` = 151,643 (the tokenizer's
   base vocabulary). So the draw is from 0 to 151,641, and added special tokens such as the mask token
   are never drawn. If the draw equals the original token, it is redrawn up to 5 times. If all six
   draws collide (probability 151,642⁻⁶ ≈ 8 × 10⁻³² per replaced token), the token becomes
   (original + r) mod `vocab_size` with r uniform in 0 … 151,642. That fallback can in principle
   return id 151,642 or the original token itself.
3. **Targets.** The target at every position is the original token. The main loss keeps targets at
   masked and replaced positions only.

With `mixture_prob` = 0, `DataCollatorWithPositionIDsMasking` is used instead. It does the same masking,
replaces nothing, and keeps targets at masked positions only.

### Loss

Implemented in `Qwen2ForCausalLM.forward` (`veomni/models/transformers/qwen2/modeling_qwen2.py`).
Open-dCoder keeps the autoregressive shift: the output at position i − 1 predicts token i. As a
result, the first token of each packed chunk is never a target.

The notation, per packed micro-batch:

- ℓ_i is the cross-entropy of the original token x_i.
- S is the set of target positions that are masked or replaced.
- N ⊆ S is the set of replaced positions.
- C is the set of the remaining (clean, visible) positions.
- t_i is the mask ratio of the chunk that contains i.
- sg is stop-gradient.

The loss is

```
L =  (1/|S|) Σ_{i∈S} ℓ_i                              # "mdm"
   + (1/|S|) Σ_{i∈S} sg(exp(−ℓ_i)) · ℓ_i / t_i        # "path"
   + noise_token_wt · (1/|N|) Σ_{i∈N} ℓ_i             # "noise"
   + clean_token_wt · (1/|C|) Σ_{i∈C} ℓ_i             # "clean"
```

The names in comments are the loss components printed in the training log. Details:

- **Normalisation.** Every term is a token mean over its set, pooled over all chunks in the
  micro-batch. It is not a per-sequence mean. Each denominator gets +1e-8.
- **1/t weighting.** The masked-position term ("mdm") has no 1/t weighting. Only the "path" term,
  weighted by the model's own detached probability of the target, carries 1/t.
- **Skipped terms.** The noise and clean terms are skipped when their weight is 0 or their set is
  empty.
- **Averaging.** Each micro-batch loss is divided by the number of gradient-accumulation steps. DDP
  then averages over data-parallel ranks.
- **Replaced positions enter three terms.** They are part of S, so a replaced token's cross-entropy
  gets weight 1/|S| in "mdm", the detached weight exp(−ℓ_i)/(t_i·|S|) in "path", and
  `noise_token_wt`/|N| (0.1/|N| for CDLM) in "noise".
- **MDLM.** For MDLM, S is the set of masked positions and the loss is "mdm" + "path". Both arms share
  these two terms.
- **Other logged components** (mixture runs only, computed without gradient): `noise_conf` and
  `clean_conf` are the mean probability the model assigns to the token currently at the replaced and
  at the clean visible positions, and `conf_ratio` = `clean_conf`/`noise_conf`, clamped to [0, 10000].
  The code of the headline runs clamped it at 100, so their logged `conf_ratio` saturates at 100.

### Relation to Eq. (1) in the paper

Eq. (1) as printed is (1/|M|) Σ_{i∈M} ℓ_i + λ_noise Σ_{i∈V} z_i ℓ_i, with M the masked positions,
V the visible positions and z_i = 1 for a replaced token. The implementation differs in three ways:

1. The first term ("mdm") is averaged over S = masked ∪ replaced positions, not over the masked
   positions only.
2. There is an additional stop-gradient, 1/t-weighted "path" term, inherited from Open-dLLM's loss
   and present in both arms.
3. The noise term is a mean over the replaced positions, not an unnormalised sum.

In addition, t is clamped to [1/500, 1 − 1/500], and every mean is pooled over the packed
micro-batch. The implementation, not Eq. (1), is authoritative for all reported numbers.

### 8B and Sudoku

The 8B and Sudoku experiments use their own implementations of the same idea:

- **LLaDA-8B** (`llada8b_lora/train_llada_mixture.py`) uses LLaDA's estimator Σ_masked ℓ_i / p / (B·L)
  on masked positions only, with p = (1 − 10⁻³)·t + 10⁻³, no shift and no path term, plus
  `noise_token_wt` × the mean cross-entropy on replaced positions.
- **Sudoku** (`sudoku/train.py`) uses the mean cross-entropy on masked cells plus 1.0 × the mean
  cross-entropy on replaced cells.

## Released checkpoints

The CDLM-0.5B model and the CRB data are in the
[CDLM collection](https://huggingface.co/collections/Shuibai12138/cdlm). The other checkpoints are
public repositories under the same account.

| Model | Hugging Face | Produced by |
| --- | --- | --- |
| Base (before our training) | [`fredzzp/open-dcoder-0.5B`](https://huggingface.co/fredzzp/open-dcoder-0.5B) | Open-dLLM |
| CDLM-0.5B | [`Shuibai12138/CDLM-0.5B`](https://huggingface.co/Shuibai12138/CDLM-0.5B), same weights as [`Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000`](https://huggingface.co/Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000) | step 2000 of the headline CDLM run (below); `scripts/train_0.5b.sh`, `ARM=cdlm` |
| MDLM-0.5B | [`Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000`](https://huggingface.co/Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000) | step 2000 of the headline MDLM run (below); `scripts/train_0.5b.sh`, `ARM=mdlm` |
| α sweep | `Shuibai12138/open-dcoder-ablation-<α>`, twelve checkpoints, α ∈ {0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, e.g. [`open-dcoder-ablation-0.3`](https://huggingface.co/Shuibai12138/open-dcoder-ablation-0.3) | `scripts/train_alpha_sweep.sh <α>` |
| CDLM + ctw | [`Shuibai12138/open-dcoder-ablation-0.1-ctw0.1`](https://huggingface.co/Shuibai12138/open-dcoder-ablation-0.1-ctw0.1) | `scripts/train_clean_token_ablation.sh` |
| CRB instances | [`Shuibai12138/crb-datasets`](https://huggingface.co/datasets/Shuibai12138/crb-datasets) | `codecorrection/generate.py` |

The top-level CRB pipeline picks the Open-dCoder code path only when the model name contains
`open-dcoder` (case-insensitive). To evaluate CDLM-0.5B with the pipeline, pass
`Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000`, which has the same weights, or a local copy
whose path contains `open-dcoder`. Don't pass `Shuibai12138/CDLM-0.5B`.

The `LLaDA_8B_Base` split of `crb-datasets` is not the input set of the LLaDA-8B experiment; see
[`llada8b_lora/README.md`](llada8b_lora/README.md#crb-inputs).

Not released: the seed-study checkpoints, the LLaDA-8B LoRA adapters, the Sudoku checkpoints, and
α = 0.9999, for which the sweep script accepts the value but no model was uploaded.

### Provenance of CDLM-0.5B and MDLM-0.5B

- **Runs.** CDLM-0.5B is step 2000 of run `Open-Dcoder-0.5B-continue-mixture-mdm-20251115-115315`.
  MDLM-0.5B is step 2000 of run `Baseline-Open-Dcoder-0.5B-continue-mixture-mdm-20251115-220242`.
- **Setup of both runs.** Seed 42, 4 × A100-PCIe-40GB, micro batch 3 × 4 GPUs × accumulation 1 =
  global batch 12, the long-horizon schedule below. Environment: Python 3.11.13, torch 2.5.0
  (CUDA 12.4 build), transformers 4.54.1, tokenizers 0.21.4, datasets 3.6.0, every other package as
  in `requirements.txt`.
- **Code.** The runs used commit 409879f of the development repository. This snapshot is its later
  commit 4c13f87. The four files that changed in between affect only an export path, timeouts,
  comments and the logged `conf_ratio` clamp, not the data, loss or gradients ([`NOTICE`](NOTICE)).
- **Evidence.** Loaded into this code, in the `requirements.txt` environment, and trained with
  lr = 1e-30 on the headline data stream, the published weights give a step-2001 loss bitwise
  equal to the one each original run logged at step 2001. Every loss component matches too (for
  CDLM, all except the `conf_ratio` clamp).
- **Configurations.** [`configs/cdlm/headline_resolved.yaml`](configs/cdlm/headline_resolved.yaml)
  (CDLM) and [`headline_mdlm_resolved.yaml`](configs/cdlm/headline_mdlm_resolved.yaml) (MDLM) are the
  configurations the trainer saved, with paths, run name and W&B entity replaced by placeholders.
  They differ only in `mixture_prob` and `noise_token_wt`. They are for reference only: veomni's
  parser reads `null` and `[]` back as strings, so `tasks/train_torch.py` cannot load them.
- **Relation to `train_0.5b.sh`.** `ARM=cdlm|mdlm SEED=42 NPROC=4 bash training/scripts/train_0.5b.sh`
  resolves to the same configuration except `save_steps` (10000 → 2000),
  `save_time_interval_minutes` (170 → 0), `eval_every` (1000 → 0) and `eval_before_train`
  (true → false). The original `train_path` was one directory read in `os.listdir` order; the
  launcher uses the recorded order instead (see [Data](#data)).
- **Not retrained.** No 2000-step run of the release scripts has been compared with the released
  checkpoints.

## Environment

The 0.5B runs used Python 3.11 with torch 2.5.0 (CUDA 12.4 build), flash-attn 2.7.4.post1 and
liger-kernel 0.5.8 on Linux x86_64:

```bash
conda create -n cdlm-train python=3.11 -y && conda activate cdlm-train
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install ninja packaging
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r training/requirements.txt
```

**Which runs used which versions.** The pins in `requirements.txt` are the environment of the
seed-study runs (seeds 1234 and 2025) and the CDLM + ctw run, which used exactly this code. The
headline seed-42 runs behind CDLM-0.5B and MDLM-0.5B used the same versions except
transformers 4.54.1, tokenizers 0.21.4 and datasets 3.6.0. The pinned environment reproduces the
headline data stream bitwise: a fresh MDLM run with `train_0.5b.sh` matches the recorded losses of
steps 1 and 2, and the frozen replay above matches step 2001 for both arms. datasets 3.6.0 and 4.2.0
therefore give the same stream here. To use the headline versions instead, run
`pip install transformers==4.54.1 tokenizers==0.21.4 datasets==3.6.0` after `requirements.txt`.
That combination is consistent with the other pins but was not re-run with this snapshot.

`requirements.txt` has more notes. The points that matter most:

- **flash-attn.** It downloads a prebuilt wheel when one matches the environment. Otherwise it
  compiles, which needs `nvcc` and a lot of RAM (set `MAX_JOBS=4`).
- **liger-kernel.** It is required; the loss raises an error without it. Its triton kernels are
  compiled at run time, which needs a C compiler (or `CC`) and the Python headers (conda Python
  ships them).
- **Pillow and setuptools.** Both are required. Every `import veomni.models` (trainer and
  converter) reaches a module that needs Pillow, and triton 3.1.0 imports setuptools at start-up
  without declaring it (bare venvs lack it).
- **transformers.** 4.57.0 was later yanked on PyPI; the exact pin still installs, with a warning.

The LLaDA-8B experiment has its own `llada8b_lora/requirements.txt` (adds `peft`) and Sudoku has
`sudoku/requirements.txt`.

`training/veomni` and the top-level `Open-dLLM/veomni` are both importable as `veomni`, and they
differ. The launch scripts put `training/` first on `PYTHONPATH` for the training process only.
Never put `training/` on `PYTHONPATH` for the CRB evaluation, which imports `veomni` from
`Open-dLLM/` when it loads an Open-dCoder model:

```bash
# training: nothing to set; the launchers do it
bash training/scripts/train_0.5b.sh
# evaluation, from the repository root
export PYTHONPATH="$PWD/Open-dLLM"
python -c "import veomni; print(veomni.__file__)"   # should print <repo>/Open-dLLM/veomni/__init__.py
```

**W&B.** Logging defaults to offline (`WANDB_MODE=offline`, logs under the run directory). To log
online, run `wandb login` (or set `WANDB_API_KEY`), then set `WANDB_MODE=online` and optionally
`WANDB_ENTITY` and `WANDB_PROJECT`. The launchers always pass `--train.wandb_entity`, empty when
`WANDB_ENTITY` is unset, because the config's `wandb_entity: null` would otherwise reach W&B as the
entity "null". The W&B run id is the run name, with `resume="allow"`. The default `RUN_NAME` of
`train_0.5b.sh` is deterministic, so a second run with the same name in the same entity and project
continues the earlier W&B run, and W&B ignores the new run's steps up to the earlier run's last
one. Use a new `RUN_NAME` per attempt. The α-sweep and ctw scripts put a timestamp in the name.
Online logging has not been tested against the W&B server.

## Data

All 0.5B runs train on the Nemotron-SFT-Code subset of the gated dataset
[`nvidia/Nemotron-Pretraining-SFT-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1)
(78 parquet shards, 76.4M rows, 59.3 GB). The subset is used as-is, with no preprocessing. To
download it, first accept the dataset terms on the Hub, then run:

```bash
huggingface-cli login
python training/data_prep/prepare_nemotron_sft_code.py --paper_order   # add --sha256 to hash every shard
```

This downloads the pinned revision to `data/Nemotron-SFT-Code/` at the repo root (override with
`--local_dir` or `DATA_ROOT`) and checks file names and sizes. That directory is the default
`DATA_DIR` of every script in `scripts/`.

The veomni loader lists the data directory with `os.listdir()`, so the order in which shards are
streamed depends on the filesystem. `--paper_order` writes `data/train_path_paper_order.txt`, which
records the order of the machine that ran the A100 runs. The three launchers use it automatically:

- **Default (`PAPER_ORDER=1`).** If `TRAIN_PATH` is unset and `DATA_DIR/../train_path_paper_order.txt`
  exists, the shards are read in that order. The launcher first checks that the file lists exactly
  the shards of `DATA_DIR`, and stops with the command to regenerate it if not.
- **Opting out.** `PAPER_ORDER=0` uses the `os.listdir` order of `DATA_DIR`, which is also what you
  get when the file does not exist.
- **Explicit order.** `TRAIN_PATH` (comma-separated absolute shard directories) overrides both;
  `TRAIN_PATH="$(cat data/train_path_paper_order.txt)"` gives the same arguments as the default.

Every launcher prints a `Shard order:` line saying which of these applies. See
[`data_prep/README.md`](data_prep/README.md) for the details and for what was verified.

The LLaDA-8B trainer reads the same shards from the same default location, so one download serves
both.

### OpenCodeInstruct variant

Nemotron-SFT-Code is gated and cannot be redistributed. For uses that need public training data,
[`scripts/train_0.5b_opencodeinstruct.sh`](scripts/train_0.5b_opencodeinstruct.sh) trains with the
CDLM-0.5B settings on
[`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (CC BY 4.0, not
gated) instead:

```bash
python training/data_prep/prepare_opencodeinstruct.py          # no login; 6.9 GB download, 2.7 GB rendered
bash training/scripts/train_0.5b_opencodeinstruct.sh           # ARM, SEED, NPROC, ... as in train_0.5b.sh
```

**This is not the paper's data. Models trained on it are not the paper's models, and their results
are not the paper's results.** No released checkpoint and no reported number comes from it. See
[`data_prep/README.md`](data_prep/README.md#ungated-substitute-opencodeinstruct) for how the data is
rendered and how it differs from the paper corpus.

## Scripts and paper experiments

All commands run from the repository root. The scripts find their own paths, create the run
directory under `OUTPUT_DIR` (default `training/outputs/`) and print every hyperparameter before
starting.

| Paper experiment | Command | Settings | LR regime |
| --- | --- | --- | --- |
| CDLM-0.5B results in Secs. 6–7 and Apps. C and E (e.g. Figs. 4–5, the CDLM row of Table 3 and the "CDLM (ours)" row of Table 4) | `ARM=cdlm bash training/scripts/train_0.5b.sh` | α 0.1, noise wt 0.1, seed 42, 4 GPUs | long horizon, stopped at step 2000 |
| The matched MDLM-0.5B baseline in the same results (e.g. Figs. 4–5, App. C) | `ARM=mdlm bash training/scripts/train_0.5b.sh` | α 0, noise wt 0, seed 42, 4 GPUs | long horizon, stopped at step 2000 |
| Seed study: seeds 1234 and 2025, both arms (added in the camera-ready version, NeurIPS 2026) | `ARM=cdlm SEED=1234 NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash training/scripts/train_0.5b.sh` (repeat for `mdlm` and `SEED=2025`) | as above, 2 GPUs × grad accum 2 | long horizon, stopped at step 2000 |
| Effect of mixture probability (Fig. 6) | `bash training/scripts/train_alpha_sweep.sh <α>` | α as below, noise wt 0.1 | `max_steps=2000` |
| Clean-token supervision, "CDLM + ctw" row (Table 4) | `bash training/scripts/train_clean_token_ablation.sh` | α 0.1, noise wt 0.1, clean wt 0.1 | `max_steps=2000` |
| Transfer to LLaDA-8B-Base (added in the camera-ready version, NeurIPS 2026) | `bash training/llada8b_lora/run_train.sh {mdlm,cdlm} 2000 full` | LoRA r 64; see [`llada8b_lora/README.md`](llada8b_lora/README.md) | own cosine schedule (below) |
| Sudoku, Appendix F (Figs. 7–9) | `bash sudoku/scripts/run_all.sh` | from scratch; see [`sudoku/README.md`](../sudoku/README.md) | own cosine schedule (below) |

Notes on the mapping:

- **Camera-ready rows.** See the note at the top of this file.
- **α values.** Fig. 6 plots the nine values α = 0.1, 0.2, …, 0.9. Twelve α checkpoints are on the
  Hub: those nine plus α = 0.04, 0.06 and 0.08. The submission's text speaks of 10 models.
- **Table 3.** Its "Open-dCoder-0.5B" row is the untrained base model `fredzzp/open-dcoder-0.5B`
  (0.143 at τ = 0.9, as in Table 1), not MDLM-0.5B.

Shared 0.5B settings (the config plus the launcher flags):

| Setting | Value |
| --- | --- |
| Base model | `fredzzp/open-dcoder-0.5B`, flash-attention 2 |
| Sequences | packed 4096-token sequences, iterable (streaming) dataset |
| Batch | micro batch 3, global batch 12 (3 × 4 GPUs, or 3 × 2 GPUs × accumulation 2), about 49K token positions per step |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, gradient clipping 1.0 |
| Schedule | cosine decay to 3e-6, `lr_warmup_ratio` 0.001 |
| Parallelism | DDP, `--train.ckpt_manager=dcp` |

### Learning-rate schedules

The two 0.5B regimes differ only in whether `--train.max_steps` is passed. They give very different
learning rates over the first 2000 steps:

- **Long horizon, stopped at step 2000** (`train_0.5b.sh`: headline CDLM/MDLM and the seed study).
  - No `max_steps` is passed, so veomni derives the schedule length from `train_size` = 1e12 tokens:
    train_steps = ⌈1e12 / (12 × 4096)⌉ = 20,345,053, and warmup = ⌊0.001 × train_steps⌋ = 20,345
    steps.
  - The run is stopped right after the step-2000 checkpoint is saved (see
    [How `train_0.5b.sh` stops](#how-train_05bsh-stops)). It is therefore still in linear warmup:
    lr(2000) = 3e-4 × 2000 / 20,345 ≈ 2.95e-5, about 10% of the peak.
- **`max_steps=2000`** (`train_alpha_sweep.sh`, `train_clean_token_ablation.sh`).
  - The schedule is exactly 2000 steps long: 2 warmup steps, then a full cosine decay from 3e-4 to
    3e-6 at step 2000.

Checkpoints from different regimes are not interchangeable:

- The α = 0.1 point of the sweep is not the headline CDLM-0.5B checkpoint.
- In Table 4, the "CDLM (ours)" row reports the headline checkpoint (long horizon), while "CDLM + ctw"
  was trained with `max_steps=2000`. Steps, data and batch size match, but the LR schedule does not.

The LLaDA-8B runs use a third schedule: peak 3e-4, 200 warmup steps, cosine over a 20,000-step
horizon, stopped at step 2000 (lr ≈ 2.94e-4). The Sudoku runs use cosine from 1e-4 to 1e-5 over
200,000 steps with no warmup.

### Useful overrides

The `scripts/` launchers read these environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `NPROC` | 1, 2 or 4; the global batch stays 12 | 4 |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0..NPROC-1` |
| `MASTER_PORT` | rendezvous port | 29500 (`train_0.5b.sh`), the original per-α port (sweep), 29400 (ctw) |
| `DATA_DIR` | parquet directory | `data/Nemotron-SFT-Code` |
| `PAPER_ORDER` | 1: read the shards in the order of `DATA_DIR/../train_path_paper_order.txt` when it exists; 0: `os.listdir` order | 1 |
| `TRAIN_PATH` | comma-separated absolute shard directories, passed verbatim; overrides `DATA_DIR` and `PAPER_ORDER` | unset |
| `OUTPUT_DIR` | where run directories are created | `training/outputs` |
| `BASE_MODEL` | Hub id or local directory; a relative path is relative to the directory you run the script from | `fredzzp/open-dcoder-0.5B` |
| `WANDB_MODE` | `offline`, `online` or `disabled` | `offline` |
| `WANDB_PROJECT` | W&B project | `Qwen2.5-Coder-0.5B` (`train_0.5b.sh`), `mixture_ablation` (the other two) |
| `WANDB_ENTITY` | W&B entity | unset: the account's default entity |

`train_0.5b.sh` additionally takes:

| Variable | Meaning | Default |
| --- | --- | --- |
| `ARM` | `cdlm` or `mdlm` | `cdlm` |
| `SEED` | random seed | 42 |
| `STOP_STEP` | step to save at and stop | 2000 |
| `RUN_NAME` | name of the run directory, also the W&B run name and id; must not contain `/ : ; , # ? '` | `open-dcoder-0.5B-<ARM>-seed<SEED>-step<STOP_STEP>` |
| `PRUNE_DCP` | 1 deletes the ~2.6 GB of dcp shards after the HF export | 0 |
| `STOP_GRACE_SECONDS` | seconds between SIGINT and SIGKILL when stopping | 60 |
| `POLL_SECONDS` | how often the log is checked for the checkpoint | 10 |

`train_0.5b.sh` refuses to start in a run directory that already has checkpoints, because the config
enables `auto_resume`.

### How `train_0.5b.sh` stops

The script sets `save_steps = STOP_STEP` and watches `train.log` for the line
`Huggingface checkpoint saved at <run>/checkpoints/global_step_2000/hf_ckpt_step_0002000 successfully!`,
which is logged once the export is complete. It then sends SIGINT to torchrun, which forwards it to
the workers so that they exit normally and W&B can write its last steps. Processes of the run still
alive after `STOP_GRACE_SECONDS` get SIGKILL.

- A few more optimizer steps may run between the checkpoint and the stop. The saved checkpoint is
  unaffected.
- The end of `train.log` then shows torchrun's `Received Signals.SIGINT death signal` warnings and
  `KeyboardInterrupt` tracebacks. They are expected.
- `train.log` is the authoritative record. W&B completeness is best effort: a rank blocked in a
  collective is killed by torchrun after 30 s, and its last offline steps can be lost.
- Ctrl-C, SIGTERM or a hangup sent to the script stops the run the same way. A second Ctrl-C during
  the grace period kills at once.
- It needs `setsid` (util-linux).

## Converting checkpoints to HuggingFace format

Training already exports each saved checkpoint in HuggingFace format:

- `train_0.5b.sh`: `<run>/checkpoints/global_step_2000/hf_ckpt_step_0002000`
- max_steps scripts: `<run>/checkpoints/global_step_2000/hf_ckpt`

`tools/convert_to_hf.sh` exposes that export under a name the CRB pipeline recognises. If the export
is missing, it converts the dcp shards with `tools/convert_dcp_to_hf.py` (CPU, a few seconds):

```bash
bash training/tools/convert_to_hf.sh training/outputs/open-dcoder-0.5B-cdlm-seed42-step2000
bash training/tools/convert_to_hf.sh training/outputs/mixture-mdm-mp0.3-ntw0.1-<timestamp> open-dcoder-ablation-0.3
# -> training/outputs/models/<name>  (MODE=copy to copy instead of symlink, FORCE_CONVERT=1 to convert the shards)
```

`open-dcoder-0.5B-` is prepended to the name unless it already contains `open-dcoder`.

## Evaluating with the CRB pipeline

Trained 0.5B models plug into the top-level pipeline (`codecorrection/generate.py`, `refine_code.py`,
`evaluate_code.py`) like any other `--model_name`. Use the evaluation environment from the main
[Quick Start](../README.md#quick-start) with `PYTHONPATH="$PWD/Open-dLLM"` (see
[Environment](#environment)), so that `veomni` comes from `Open-dLLM/`.

The two `modeling_qwen2.py` copies differ only in `Qwen2ForCausalLM`: the training losses, their
logging and an unused teacher-alignment path. The forward pass used for inference is the same. All
Open-dCoder-derived checkpoints share the base tokenizer, so a single set of CRB inputs, generated
with `fredzzp/open-dcoder-0.5B`, serves all of them.

```bash
# 1. corrupt and pre-evaluate (once per dataset / error type / n_replace)
python codecorrection/generate.py --dataset human-eval --error_type operator --n_replace 1 --data_num 2 \
    --model_name fredzzp/open-dcoder-0.5B --data_path buggy_datasets --deduplicate
python evaluate_code.py --results_file buggy_datasets/human-eval/open-dcoder-0.5B_operator_2_wrong_1.jsonl \
    --output_file buggy_datasets/human-eval/evaluated/open-dcoder-0.5B_operator_2_wrong_1_evaluated.jsonl \
    --dataset human-eval --map_prompt2completion --no_postprocess

# 2. refine with a trained model (T refinement steps = --refined_steps - 1; tau = 0.9)
torchrun --nproc_per_node=1 --master_port=29502 refine_code.py \
    --initial_results_file buggy_datasets/human-eval/evaluated/open-dcoder-0.5B_operator_2_wrong_1_evaluated.jsonl \
    --model_name training/outputs/models/open-dcoder-0.5B-cdlm-seed42-step2000 \
    --batch_size 1 --refined_steps 2 --algorithm self_conf-remask:vanilla --temperature 0.0 \
    --refine_setting remove_all --confidence_threshold 0.9 --output_prefix cdlm

# 3. score the refined programs
REFINED=$(find cdlm_results -name '*_results_refined.jsonl' | head -n 1)
python evaluate_code.py --results_file "$REFINED" --output_file "${REFINED%.jsonl}_evaluated.jsonl" \
    --dataset human-eval --no_postprocess
```

Some notes on running these steps:

- **Separate outputs per model.** `--output_prefix` keeps each model's outputs apart. The default,
  `correction`, is shared by all models.
- **Released checkpoints.** To use a released checkpoint instead, pass e.g.
  `--model_name Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000`.
- **Reference script.** `examples/test_human-eval_open-dllm.sh` runs the same steps end to end.
- **var and literal inputs.** `codecorrection/generate.py` now gives the same files for a given
  `--seed` on every run. Earlier versions made the `var` and `literal` corruptions depend on
  Python's hash seed, so files generated with them, including the inputs of the paper's
  experiments, cannot be regenerated exactly. `operator` files are unaffected.
- **LLaDA-8B.** The LoRA adapters are evaluated through `llada8b_lora/refine_code_lora.py`; see
  [its README](llada8b_lora/README.md).

## Hardware and runtime

| Run | Hardware used for the paper | Time | Disk per run |
| --- | --- | --- | --- |
| CDLM / MDLM-0.5B, seed 42 | 4 × A100-PCIe-40GB, DDP | about 30 min to step 2000 (about 0.8 s/step) | about 3.8 GB (1.2 GB HF export plus dcp shards) |
| Seed study | 2 × A100-PCIe-40GB, grad accum 2 | 42–52 min to step 2000 (1.26–1.57 s/step) | about 3.8 GB |
| α sweep | 4 × H200 | not recorded | about 12 GB, because the in-training eval hook saves extra checkpoints |
| CDLM + ctw | 4 × A100-PCIe-40GB | about 30 min | about 13 GB |
| LLaDA-8B LoRA | 4 × A100-PCIe-40GB, about 24 GB peak per GPU | about 46 min per 2000-step arm; the full CRB sweep takes about 1 h 50 min on 4 GPUs | 640 MB adapter |
| Sudoku | 1 GPU per run | training: not recorded; evaluation: 1,350 evaluation processes, about 6–14 h for both checkpoints run one after another on A100-PCIe-40GB GPUs (the two sweeps can run in parallel on two GPUs); see [`sudoku/README.md`](../sudoku/README.md#runtime) | about 10 GB of checkpoints per run |

The 4-GPU 0.5B time comes from the log of the clean-token run, which used the same hardware and batch
size. The seed-study time comes from the logs of those runs.

## Reproducibility notes

- **Not bitwise reproducible over a whole run.** `enable_full_determinism` is false, and the
  flash-attn and liger backward passes are nondeterministic. In our checks, release runs matched the
  recorded losses bitwise at steps 1 and 2 and differed slightly afterwards.
- **What else determines the data stream.** Besides the seed, the data stream depends on:
  - the shard order (see [Data](#data));
  - the number of data-parallel ranks, because data is sharded per rank.

  Use `NPROC=4` for the seed-42 runs and `NPROC=2` for seeds 1234 and 2025. The α sweep ran on a
  different machine, whose shard order was not recorded; the α-sweep launcher uses the headline
  runs' recorded order.
- **Base model revision.** `fredzzp/open-dcoder-0.5B` is loaded without a pinned revision, i.e. from
  the Hub's current `main`.
- **Differences from the original headline runs.** They are listed under
  [Provenance](#provenance-of-cdlm-05b-and-mdlm-05b): checkpointing and evaluation flags only. The
  step-2001 replay ran without the evaluation hook and still matched both headline runs bitwise, so
  the hook's evaluation steps did not shift the data stream.
- **In-training HumanEval hook.** `train_torch.py` runs this hook at every checkpoint save, and also
  at `eval_before_train` and every `eval_every` steps; `--train.eval_every=0` does not disable it. The
  hook runs `python eval/eval_completion/eval_single.py` from `training/`, which is not shipped here,
  so the log shows

  ```
  python: can't open file '<repo>/training/eval/eval_completion/eval_single.py': [Errno 2] No such file or directory
  ```

  followed by `results_files: []` and a warning `No results files found matching pattern: ...`. The
  message is harmless: the failure is ignored and training continues. `train_0.5b.sh` prints it once,
  right after the step-`STOP_STEP` checkpoint message. `train_alpha_sweep.sh` and
  `train_clean_token_ablation.sh` keep the original `eval_before_train=true` and `eval_every=1000`,
  so it appears three times (before training, at steps 1000 and 2000), and these evaluation steps
  save extra checkpoints, about 8 GB per run.

## License

`veomni/`, `tasks/` and `configs/` are derived from [VeOmni](https://github.com/ByteDance-Seed/VeOmni)
and [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) and remain under the Apache License 2.0
([`LICENSE`](LICENSE), attribution and list of changes in [`NOTICE`](NOTICE)). The rest of the
repository is MIT-licensed; `sudoku/puzzle_generator.py` and `sudoku/advanced_sudoku_generator.py`
are third-party code by Ali Alp (MIT, per the upstream README; see
[`sudoku/THIRD_PARTY_NOTICES.md`](../sudoku/THIRD_PARTY_NOTICES.md)). The training data and the base
models are subject to their own licenses (for Nemotron-SFT-Code, the terms you accept on its Hub
page; see [`data_prep/README.md`](data_prep/README.md)).
