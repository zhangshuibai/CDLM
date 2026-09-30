# LLaDA-8B-Base LoRA corrective training

This directory contains the code for the 8B transfer experiment: a controlled, compute-matched
comparison between absorbing-only masked-diffusion fine-tuning (**MDLM**) and the corrective
mixture objective (**CDLM**) on `GSAI-ML/LLaDA-8B-Base`. The two arms use the same corpus, steps,
schedule, seed, batch and LoRA configuration. Only `mixture_prob` and `noise_token_wt` differ.

This experiment is not in the NeurIPS 2026 submission. It is added in the camera-ready version
(NeurIPS 2026), and its numbers were also reported in the public OpenReview discussion of the
submission.

## Reported results

Localisation is measured on CRB HumanEval with `n_replace=1` and all three error types (n = 541).
Confidence is the probability the model assigns to the token currently at each position,
`gather(softmax(logits), -1, x)`, which is the same quantity the `self_conf-remask:vanilla`
sampler uses. Pass@1 is at T=1 (`--refined_steps 2`) with τ = 0.9, `remove_all`,
temperature 0 and batch size 1. It is macro-averaged over {HumanEval, HumanEval+, MBPP, MBPP+} ×
{operator, var, literal} with `n_replace=1`.

| model (2000 steps) | conf. gap | Top-1 hit | Top-3 hit | Pass@1 T=1, de-fenced | Pass@1 T=1, raw |
|---|---|---|---|---|---|
| LLaDA-8B-Base (no fine-tuning) | 0.265 | 16.5 | 54.7 | 55.1 | 55.1 |
| + MDLM LoRA (`mixture_prob=0`, `noise_token_wt=0`) | 0.402 | 15.2 | 73.4 | 60.0 | 47.5 |
| + CDLM LoRA (`mixture_prob=0.1`, `noise_token_wt=0.1`) | **0.777** | **60.8** | **92.8** | **72.5** | 53.0 |

Early checkpoint (a separate 200-step run with the same configuration, i.e. a prefix of the
2000-step schedule):

| model (200 steps) | conf. gap | Top-1 hit | Top-3 hit | Top-5 hit |
|---|---|---|---|---|
| MDLM | 0.258 | 10.5 | 55.6 | 80.6 |
| CDLM | 0.790 | 53.0 | 87.4 | 97.0 |

**Raw vs. de-fenced.** Both fine-tuned arms learn markdown code fences from Nemotron-SFT-Code.
The HumanEval+ buggy bodies end with a spare blank line. With `remove_all` refinement the models
fill it with "```", which makes the program a syntax error under `--no_postprocess`. `defence.py`
truncates each completion at the first "```", strips trailing whitespace, and applies the same rule
to every arm. Both numbers are reported above. At T=4 (`--refined_steps 5`) the de-fenced means
are 59.5 / 60.9 / 62.3 (base / MDLM / CDLM) and the raw means are 59.5 / 46.0 / 51.8.

- **Base.** No base completion contains a fence (0 of 6,368 in the 24 base cells), so for base the
  rule only strips trailing whitespace. Individual base cells still move slightly between the two
  evaluations (e.g. HumanEval+ operator 61.6 → 61.2 and literal 82.4 → 83.0 at T=1), within the
  evaluation noise noted below; the base means are the same.
- **CDLM at T=4.** De-fencing does not recover CDLM's HumanEval+ cells at T=4 (35.3–38.8), although
  the docstring of `defence.py` says the substring rule does; the recorded outputs match the
  current rule.
- **Printed count.** `defence: N/M completions truncated` also counts completions whose only change
  is a stripped trailing newline, so it does not measure how many fences were removed.

## Released adapters

The adapters of the reported runs are on the Hugging Face Hub. Each repository holds the PEFT adapter
(`adapter_config.json`, `adapter_model.safetensors`), the trainer's `train_config.json` and
`train_log.json`, and a model card; the 200-step runs are in the `step200/` subfolder.

| adapter | Hugging Face | revision |
|---|---|---|
| CDLM LoRA, 2000 steps (200 steps: `step200/`) | [`Shuibai12138/LLaDA-8B-CDLM-LoRA`](https://huggingface.co/Shuibai12138/LLaDA-8B-CDLM-LoRA) | `84fb1d238fc313e17eb1541dd41d74f719670554` |
| MDLM LoRA, 2000 steps (200 steps: `step200/`) | [`Shuibai12138/LLaDA-8B-MDLM-LoRA`](https://huggingface.co/Shuibai12138/LLaDA-8B-MDLM-LoRA) | `97cb772d4555a35b33186d2383737fe37abacc89` |

Every script here that takes an adapter accepts a local directory or a Hub id
`<owner>/<name>[/<subfolder>][@<revision>]` (resolved by `adapter_path.py`). Loaded from the Hub
without a token, the four adapters reproduce the recorded localisation results of the table above
exactly (every per-file and pooled value); this was checked at the first upload (`eea470e0`,
`d9f1ca8f`), whose adapter files are the same as at the listed revisions (later commits changed only
the model cards). The 2000-step adapter files are those the
recorded CRB sweep and localisation loaded, and the `step200/` files are those the recorded 200-step
localisation loaded. None of them was modified after training.

## Files

| file | purpose |
|---|---|
| `train_llada_mixture.py` | Standalone trainer: HF `AutoModel` + `peft` LoRA + torchrun DDP. The corruption process is `forward_process` (L111) and the loss is `compute_loss` (L170). |
| `run_train.sh` | Launches one arm with the hyperparameters of the reported runs. |
| `run_train_opencodeinstruct.sh` | The same launch on the public nvidia/OpenCodeInstruct corpus (see [OpenCodeInstruct adapters](#opencodeinstruct-adapters)). |
| `refine_code_lora.py` | Wraps the repo-level `refine_code.py`. It loads a LoRA adapter, merges it into the bf16 base and then runs the unchanged CRB refinement. |
| `adapter_path.py` | Pins the base-model revision (`BASE_REVISION`) and resolves an adapter argument: a local directory or a Hub id with an optional subfolder and revision. |
| `run_crb_cell.sh` | Runs one CRB cell (arm × dataset × error type × n_replace × steps) on one GPU: refinement, then raw and de-fenced evaluation with the repo-level `evaluate_code.py`. |
| `run_crb_parallel.sh` | Runs the full 72-cell sweep (3 arms × 4 datasets × 3 error types × {2, 5} steps) with one queue per GPU; `ARMS` selects other arms. |
| `defence.py` | The markdown-fence stripping rule (see above). |
| `aggregate_crb.py` | Builds the raw and de-fenced Pass@1 tables from the sweep outputs (`--labels` for other arms). |
| `eval_confidence.py`, `run_conf_eval.sh` | Measure the confidence gap and Top-K localisation hit rate (K = 1..6). |
| `fetch_crb_inputs.sh`, `crb_inputs.md5` | Download the 24 CRB input files of this experiment and check their md5. |
| `requirements.txt` | The package versions used for the experiment. |

## Training objective

Both arms train on the LLaDA absorbing loss. For each sequence the trainer draws
`t ~ U(0,1)` and masks every token independently with probability `p = (1 − 1e-3)·t + 1e-3`. The
loss is `L_absorb = Σ_masked CE / p / (B·L)`. LLaDA is bidirectional, so no logits are shifted.

CDLM also corrupts visible tokens. After masking, each remaining visible token is replaced with
probability `mixture_prob` by a uniformly sampled token that is neither the mask token nor the
original token. Those positions are supervised towards the original token:
`L = L_absorb + noise_token_wt · mean_{replaced} CE`. Clean visible tokens are never supervised
(`clean_token_wt = 0`). Absorbing noise and mixture noise come from separate RNG streams, so the
masking pattern is identical in both arms.

## Hyperparameters (both arms)

| knob | value |
|---|---|
| base model | `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` (8,015,581,184 params, 8.18 B with the LoRA parameters; mask id 126336) |
| corpus | Nemotron-SFT-Code: `nvidia/Nemotron-Pretraining-SFT-v1`, config `Nemotron-SFT-Code`, 78 parquet shards |
| packing | tokenize `text`, append EOS, split into ≤ 4096-token chunks (chunks < 16 tokens dropped). Shards are shuffled with seed 42 and assigned round-robin to ranks. |
| max_seq_len | 4096 |
| batch | micro 1 × grad-accum 3 × 4 GPUs = global 12 |
| steps | 2000 (early checkpoint: a separate 200-step run) |
| optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.01 (not applied to 1-D params), grad clip 1.0 |
| LR schedule | peak 3e-4, linear warmup for 200 steps (1% of the horizon), cosine decay over a **20,000-step horizon**. LR at step 2000 is 2.94e-4 (98.0% of peak). |
| seed | 42 |
| LoRA | r = 64, α = 128, dropout 0.05 on `q_proj, k_proj, v_proj, attn_out, ff_proj, up_proj, ff_out` inside `transformer.blocks.*` only. 167.77 M trainable params (2.05%), 448 tensors. |
| frozen | everything else, including `transformer.wte` (embeddings) and `transformer.ff_out` (lm_head) |
| precision | bf16 base weights; fp32 LoRA params and AdamW states |
| activation checkpointing | LLaDA `whole_layer` |
| MDLM arm | `--arm mdlm`, which gives `mixture_prob=0`, `noise_token_wt=0`, `clean_token_wt=0` |
| CDLM arm | `--arm cdlm`, which gives `mixture_prob=0.1`, `noise_token_wt=0.1`, `clean_token_wt=0` |

`run_train.sh` does not pass `--mixture_prob` or `--noise_token_wt`. The trainer fills them in
from `--arm`. You can pass them explicitly to `train_llada_mixture.py` to run other values.

The trainer (`train_llada_mixture.py`), the localisation script (`eval_confidence.py`) and the
refinement wrapper (`refine_code_lora.py`) load `GSAI-ML/LLaDA-8B-Base` at revision
`0f2787f2d87eac5eed8a087d5ecd24277e6255b2` (`adapter_path.BASE_REVISION`), so neither the weights
nor the model's remote code can change under a run. This revision has been the Hub's `main` since
2025-10-21; every reported run loaded it, and all six released adapters record it in
`adapter_config.json`. `--model_revision` selects another revision; a local directory passed as
`--model_path` / `--model_name` is used as is. The pin was added in `v1.0.3-corrective-training`;
earlier tags loaded the Hub's current `main`, which was the same revision.

This LR schedule is the 8B experiment's own. It does not reproduce the 0.5B runs, which were
still in linear warmup at step 2000. The schedule is identical in both 8B arms.

## Hardware and runtime

The runs used 4 × NVIDIA A100-PCIe-40GB (no NVLink) with DDP. Each GPU peaked at about 23–24 GB
allocated. Training took about 1.4 s/step, so each arm took 46 min for 2000 steps (MDLM 46.5 min,
CDLM 45.9 min) and about 4.7 min for 200 steps. Each LoRA adapter is about 640 MB.

The CRB sweep ran 4 single-GPU queues on the same machine. All 72 cells at `n_replace=1` took
about 1 h 50 min. The localisation evaluation takes about a minute per model on one GPU.

## Setup

```bash
pip install -r training/llada8b_lora/requirements.txt
```

Nemotron-SFT-Code is a gated dataset. Accept its licence on the Hugging Face page, log in with
`huggingface-cli login`, and download it with the shared script used by the 0.5B runs (about 59 GB):

```bash
python training/data_prep/prepare_nemotron_sft_code.py
```

This places the 78 shards in `data/Nemotron-SFT-Code` under the repo root, which is also this
launcher's default `DATA_DIR`. The runs reported here read a copy downloaded at Hub revision
`af7991c59eeb5e53a98bb6b1ee7a96cc3754eb39`, the revision the script pins (the download metadata of
all 78 shards records it). The Hub's later revision `3f1a5b884d0b890f02ead979ce698dc95debd953` has
the same 78 files (`part_000000.parquet` to `part_000077.parquet`) with identical content.
Point `DATA_DIR` elsewhere if you stored the data somewhere else. The trainer uses every
`*.parquet` file in that directory, so the data order depends on the shard file names and on the
number of ranks.

## Training

All commands run from the repo root. `run_train.sh <arm> <max_steps> <tag>` writes the run to
`${OUTPUT_DIR}/<tag>_<arm>/`. `OUTPUT_DIR` defaults to `outputs/llada8b_lora`. The run directory
holds `final/` (the adapter), `train_config.json`, `train_log.json` and `train.log`.

```bash
# 2000-step runs, one arm after the other on GPUs 0-3
bash training/llada8b_lora/run_train.sh mdlm 2000 full
bash training/llada8b_lora/run_train.sh cdlm 2000 full

# the 200-step early checkpoint (same schedule, so it is a prefix of the runs above)
bash training/llada8b_lora/run_train.sh mdlm 200 gate
bash training/llada8b_lora/run_train.sh cdlm 200 gate
```

These environment variables override the defaults: `CUDA_VISIBLE_DEVICES` (default `0,1,2,3`),
`NPROC` (4), `MASTER_PORT` (29642), `DATA_DIR`, `OUTPUT_DIR` and `TORCHRUN`. The global batch must
equal micro × accum × world, so `NPROC` must divide 12. Keep `NPROC=4` to get the same data order
and per-rank RNG streams as the reported runs. To change the model path or any other flag, call the
trainer directly:

```bash
torchrun --nproc_per_node=4 training/llada8b_lora/train_llada_mixture.py --arm cdlm \
    --model_path GSAI-ML/LLaDA-8B-Base --data_dir <DATA_DIR> --output_dir <OUT> \
    --max_steps 2000 --cosine_horizon 20000 --warmup_ratio 0.01 --lr 3e-4 \
    --weight_decay 0.01 --max_grad_norm 1.0 --micro_batch_size 1 --global_batch_size 12 \
    --max_seq_len 4096 --seed 42 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --clean_token_wt 0.0 --save_steps 0 --log_every 20
```

The log line is written at step 1 and every `--log_every` steps (20 in `run_train.sh`). `loss` is
the step loss (mean over the 3 micro-batches) averaged over ranks. `mdm` and `noise` are rank 0's
values only. `gnorm` is the gradient norm before clipping. `lr` is read after `scheduler.step()`,
so it is the rate the next step will use. `train_log.json` stores `step`, `loss`, `lr`, `mdm` and
`noise` the same way.

## Evaluation on CRB

### CRB inputs

The experiment reads the LLaDA-tokenised CRB files generated with `--data_num 2`. They go under
`buggy_datasets/` at the repo root:

```
buggy_datasets/<ds>/LLaDA-8B-Base_<err>_2_wrong_1.jsonl                       # localisation
buggy_datasets/<ds>/evaluated/LLaDA-8B-Base_<err>_2_wrong_1_evaluated.jsonl   # refinement
```

Here `<ds>` is one of `human-eval`, `human-eval+`, `mbpp`, `mbpp+` and `<err>` is one of
`operator`, `var`, `literal`. Download the exact 24 files of this experiment with

```bash
bash training/llada8b_lora/fetch_crb_inputs.sh
```

It takes them from the Hub dataset
[`Shuibai12138/crb-paper-inputs`](https://huggingface.co/datasets/Shuibai12138/crb-paper-inputs) at
revision `a00037635943c930fa5d8e0961c7ca26127ca53f` (the evaluated files from `llada-8b-base/`,
the raw files from `llada-8b-base-localisation/`) and checks each against `crb_inputs.md5`. No token
is needed.

- **var and literal instances differ between the two file sets.** The raw var and literal files
  were regenerated after their evaluated files had been made, and the original raw files are lost.
  The localisation evaluation reads the regenerated raw files and the repair sweep reads the
  evaluated files, so the two measure only partly the same var and literal instances. In the three
  HumanEval files the localisation reads, 46 of 136 var and 89 of 155 literal records are identical
  to evaluated records (all 250 operator records are), so 156 of its 541 records do not occur in the
  repair inputs. This is how the experiment was run; the dataset card has the details.
- **Regenerating.** The files were made with the repo's standard pipeline (see
  `examples/test_human-eval_llada.sh`):

  ```bash
  python codecorrection/generate.py --dataset <ds> --error_type <err> --n_replace 1 --data_num 2 \
      --model_name GSAI-ML/LLaDA-8B-Base --data_path buggy_datasets --deduplicate
  python evaluate_code.py --results_file buggy_datasets/<ds>/LLaDA-8B-Base_<err>_2_wrong_1.jsonl \
      --output_file buggy_datasets/<ds>/evaluated/LLaDA-8B-Base_<err>_2_wrong_1_evaluated.jsonl \
      --dataset <ds> --map_prompt2completion --no_postprocess
  ```

  For operator errors these commands (default seed 42) regenerate the four raw files byte-for-byte.
  The var and literal files cannot be regenerated: they were generated when
  `codecorrection/generate.py` still iterated a Python `set` of strings (`all_elements_text`), so
  the replacements depended on the unrecorded `PYTHONHASHSEED` of those runs. The script now
  iterates in sorted order and gives the same files for a given `--seed` on every run, but those
  files differ from the ones used here. Use the downloaded files to reproduce the numbers.
- **Hub CRB data.** The `LLaDA_8B_Base` split of
  [`Shuibai12138/crb-datasets`](https://huggingface.co/datasets/Shuibai12138/crb-datasets) is not
  this input set. For HumanEval operator with `n_replace=1` it has 944 rows against the 250 used
  here, only 13 of those 250 buggy bodies appear in it, and it uses a different schema that nothing
  in this repository converts.

### Localisation (confidence gap, Top-K hit rate)

```bash
# the released adapters
bash training/llada8b_lora/run_conf_eval.sh outputs/llada8b_lora/full_conf \
    Shuibai12138/LLaDA-8B-MDLM-LoRA@97cb772d4555a35b33186d2383737fe37abacc89 \
    Shuibai12138/LLaDA-8B-CDLM-LoRA@84fb1d238fc313e17eb1541dd41d74f719670554
# the 200-step adapters
bash training/llada8b_lora/run_conf_eval.sh outputs/llada8b_lora/gate_conf \
    Shuibai12138/LLaDA-8B-MDLM-LoRA/step200@97cb772d4555a35b33186d2383737fe37abacc89 \
    Shuibai12138/LLaDA-8B-CDLM-LoRA/step200@84fb1d238fc313e17eb1541dd41d74f719670554
# adapters you trained yourself
bash training/llada8b_lora/run_conf_eval.sh outputs/llada8b_lora/full_conf \
    outputs/llada8b_lora/full_mdlm/final outputs/llada8b_lora/full_cdlm/final
```

This evaluates base, MDLM and CDLM on HumanEval with `n_replace=1` and all three error types. It
writes `conf_{base,mdlm,cdlm}.json` with per-file and pooled `gather_gap` and `gather_hit@K`.
`LABEL_SUFFIX` (e.g. `_oci`) renames the MDLM and CDLM labels and files, and `SKIP_BASE=1` skips the
base model.
Any extra arguments are passed on to `eval_confidence.py`, for example
`--datasets human-eval mbpp` or `--n_replace 1 3`. The `max_*` fields use the
max-over-vocabulary confidence instead, which gives a different ranking.

### Pass@1 under iterative refinement

```bash
# full sweep: 3 arms x 4 datasets x 3 error types x refined_steps {2, 5}, n_replace=1
GPUS="0 1 2 3" bash training/llada8b_lora/run_crb_parallel.sh 1
python training/llada8b_lora/aggregate_crb.py --n_replace 1 --out outputs/llada8b_lora/pass1_tables.md
```

The sweep reads its adapters from `MDLM_ADAPTER` and `CDLM_ADAPTER`, which default to
`${OUTPUT_DIR}/full_{mdlm,cdlm}/final`. To use the released adapters:

```bash
MDLM_ADAPTER=Shuibai12138/LLaDA-8B-MDLM-LoRA@97cb772d4555a35b33186d2383737fe37abacc89 \
CDLM_ADAPTER=Shuibai12138/LLaDA-8B-CDLM-LoRA@84fb1d238fc313e17eb1541dd41d74f719670554 \
GPUS="0 1 2 3" bash training/llada8b_lora/run_crb_parallel.sh 1
```

`ARMS` replaces the three default arms by a whitespace-separated list of `<label>:<adapter>` pairs
(`NONE` for the base model); outputs of label `L` go to `g6v_L_results/`, and
`aggregate_crb.py --labels ...` tabulates them.

Per-GPU logs go to `${LOG_DIR}/crb_gpu<N>.log`. Refined outputs go to
`g6v_{base,mdlm,cdlm}_results/` and per-step histories go to `g6v_*_history/`, both under the repo
root. The histories can be large; you can delete them once Pass@1 has been computed.

Finished cells are reused. Each cell directory records the adapter it was produced with in
`adapter.txt` (`NONE`, or a sha256 prefix of the adapter config and weights; a Hub adapter is
downloaded first and hashed the same way, so it shares outputs with a local copy of the same files). If you
re-run a label with a different or retrained adapter, the cell stops with
`!!! STALE ... remove that directory or use another label`; delete `g6v_<label>_results` (or that
cell's directory) or use a new label. A relative adapter path is taken relative to the directory
you run the script from. A failed refinement or evaluation fails the cell, and
`run_crb_parallel.sh` then prints `FAILED cell ...` and exits non-zero. A missing input file is
reported as `SKIP` and does not fail the sweep.

Run a single cell:

```bash
bash training/llada8b_lora/run_crb_cell.sh 0 cdlm outputs/llada8b_lora/full_cdlm/final human-eval operator 1 2
```

Or call the wrapper directly. It takes the same flags as `refine_code.py` plus `--lora_adapter`.
Pass `--initial_results_file` relative to the repo root.

```bash
python training/llada8b_lora/refine_code_lora.py --lora_adapter outputs/llada8b_lora/full_cdlm/final \
    --initial_results_file buggy_datasets/human-eval/evaluated/LLaDA-8B-Base_operator_2_wrong_1_evaluated.jsonl \
    --model_name GSAI-ML/LLaDA-8B-Base --batch_size 1 --refined_steps 2 \
    --algorithm self_conf-remask:vanilla --temperature 0.0 --refine_setting remove_all \
    --confidence_threshold 0.9 --output_prefix g6v_cdlm
```

## OpenCodeInstruct adapters

Nemotron-SFT-Code is gated and licensed for internal training only. To make the 8B comparison
reproducible from public data, both arms were also trained, with the same trainer, flags and seed, on
[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (revision
`8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`, CC BY 4.0, not gated), rendered by
`training/data_prep/prepare_opencodeinstruct.py` as `"input: " + input + " output: " + output`. That is
the script of the 0.5B OpenCodeInstruct reference runs, not of the paper's 0.5B runs, which were
trained on Nemotron-SFT-Code. **These adapters are not the paper's, and no paper number comes from
them.**

```bash
python training/data_prep/prepare_opencodeinstruct.py            # download (6.9 GB) and render (2.7 GB more)
bash training/llada8b_lora/run_train_opencodeinstruct.sh mdlm    # 4 GPUs; writes outputs/llada8b_lora/oci_mdlm
bash training/llada8b_lora/run_train_opencodeinstruct.sh cdlm
```

The launcher links the 50 rendered shards into one directory, writes `data_shards.tsv` (every
shard, with the rank and position at which it is read) into the run directory, and calls
`run_train.sh`. The released adapters were trained at commit `96906ed`. Later commits added checks
and records to the launcher and pinned the base-model revision that the runs already loaded; the data
and the training arguments are unchanged. On
4 × A100-PCIE-40GB shared with other jobs, the runs took 50.8 min (MDLM) and 56.1 min (CDLM).

**What a run reads.** As in the Nemotron runs (see [Data loader](#reproducibility-notes)), each
rank reads the first 3,000 documents of one shard: shards 25, 23, 19 and 11 for ranks 0–3, i.e.
12,000 documents with 6.58 M tokens (548 per document on average). OpenCodeInstruct's rows are not
shuffled: within a shard each category comes in long contiguous runs (38 of the 50 shards hold two to
four categories), and the first 3,000 rows of each of these four shards are all of one category. These
documents are therefore not a sample of the corpus: 9,000 are
`generic`/`evol-instruct` and 3,000 `algorithmic`/`self-instruct`, while the corpus is 38.9 %
`generic`/`evol-instruct`, 33.4 % `generic`/`self-instruct`, 19.6 % `algorithmic`/`self-instruct` and
8.1 % `algorithmic`/`evol-instruct`. Both arms read the same documents.

| adapter | Hugging Face | revision |
|---|---|---|
| CDLM LoRA, OpenCodeInstruct | [`Shuibai12138/LLaDA-8B-CDLM-LoRA-OpenCodeInstruct`](https://huggingface.co/Shuibai12138/LLaDA-8B-CDLM-LoRA-OpenCodeInstruct) | `713278bc1f441641ae9fbb41b71dc87081b3bdad` |
| MDLM LoRA, OpenCodeInstruct | [`Shuibai12138/LLaDA-8B-MDLM-LoRA-OpenCodeInstruct`](https://huggingface.co/Shuibai12138/LLaDA-8B-MDLM-LoRA-OpenCodeInstruct) | `b68c4c8d8c2512d98f9ab03da3c68eebf5b1d2a1` |

Results with the protocol of [Reported results](#reported-results) (localisation on the HumanEval
files, n = 541; Pass@1 macro over the 12 cells). The base row is from the same evaluation run as the
OpenCodeInstruct rows; its localisation values equal the recorded ones exactly, and its Pass@1
values differ from them only through HumanEval/139 (see [Reproducibility notes](#reproducibility-notes)).

| model (2000 steps) | confidence gap | Top-1 hit (%) | Top-3 hit (%) | Pass@1 T=1, de-fenced | Pass@1 T=1, raw | Pass@1 T=4, de-fenced | Pass@1 T=4, raw |
|---|---|---|---|---|---|---|---|
| LLaDA-8B-Base (no fine-tuning) | 0.265 | 16.5 | 54.7 | 55.1 | 55.1 | 59.5 | 59.6 |
| + MDLM LoRA, OpenCodeInstruct | 0.338 | 10.2 | 73.0 | 57.6 | 54.6 | 59.0 | 52.1 |
| + CDLM LoRA, OpenCodeInstruct | 0.835 | 59.3 | 90.8 | 74.1 | 51.8 | 72.6 | 50.6 |
| + MDLM LoRA, Nemotron-SFT-Code (reported above) | 0.402 | 15.2 | 73.4 | 60.0 | 47.5 | 60.9 | 46.0 |
| + CDLM LoRA, Nemotron-SFT-Code (reported above) | 0.777 | 60.8 | 92.8 | 72.5 | 53.0 | 62.3 | 51.8 |

**Paired comparison.** Over the 3,184 test-failing records, CDLM-OCI repairs more programs than MDLM-OCI: de-fenced +17.0 points (p = 2e-59) at T=1 and +14.5 points (p = 4e-44) at T=4, raw +4.2 points (p = 2e-04) and +4.6 points (p = 2e-05) (pooled difference, exact McNemar test). The raw macro average nevertheless puts CDLM-OCI below MDLM-OCI, because 99.6% of its HumanEval+ completions contain a fence at T=1, so its three raw HumanEval+ cells are close to zero and weigh a quarter of the average. Without HumanEval+, the raw macro Pass@1 at T=1 is 69.1 (CDLM-OCI), 53.8 (MDLM-OCI) and 53.3 (base).

Evaluate them like the Nemotron adapters, with `ARMS` for the repair sweep:

```bash
bash training/llada8b_lora/fetch_crb_inputs.sh
SKIP_BASE=1 LABEL_SUFFIX=_oci bash training/llada8b_lora/run_conf_eval.sh outputs/llada8b_lora/oci_conf \
    Shuibai12138/LLaDA-8B-MDLM-LoRA-OpenCodeInstruct@b68c4c8d8c2512d98f9ab03da3c68eebf5b1d2a1 \
    Shuibai12138/LLaDA-8B-CDLM-LoRA-OpenCodeInstruct@713278bc1f441641ae9fbb41b71dc87081b3bdad
ARMS="base:NONE mdlm_oci:Shuibai12138/LLaDA-8B-MDLM-LoRA-OpenCodeInstruct@b68c4c8d8c2512d98f9ab03da3c68eebf5b1d2a1 cdlm_oci:Shuibai12138/LLaDA-8B-CDLM-LoRA-OpenCodeInstruct@713278bc1f441641ae9fbb41b71dc87081b3bdad" \
    GPUS="0 1 2 3" bash training/llada8b_lora/run_crb_parallel.sh 1
python training/llada8b_lora/aggregate_crb.py --n_replace 1 --labels base mdlm_oci cdlm_oci
```

## Reproducibility notes

- **Evaluation pipeline.** During the experiment, refinement ran against a frozen snapshot of the
  CRB pipeline. That snapshot's `evaluate_code.py` and `sanitize.py` are byte-identical to the
  repo-level files. Its `refine_code.py`, `llada_sample.py` and `utils.py` add batched padding,
  other baselines and a different split of samples across ranks. None of these changes affects
  LLaDA at `--batch_size 1`. A CPU check compared the repo-level pipeline with the snapshot on
  real CRB inputs, using the real LLaDA tokenizer and a small deterministic stand-in for the
  model. Collation, sampling trajectory, decoded text and history were identical in all 144
  test cases (72 samples × `--refined_steps` 2 and 5). On GPU, re-running the 24 base cells with
  the repo-level pipeline and the downloaded inputs reproduced all 6,368 refined programs of the
  recorded sweep exactly; only the test outcome of HumanEval/139 changed (it times out
  intermittently), which moves a macro Pass@1 by at most 0.09 points.
- **Batch size.** Batched bf16 inference is not bit-identical to `--batch_size 1`. Keep
  `--batch_size 1` for every arm. Sharding cells across GPUs does not change any result.
- **Evaluation noise.** Test execution sometimes times out. Identical inputs have given results
  0.4 points apart on one cell, so treat differences below about 0.5 points as noise.
- **Data loader.** The trainer's `DataLoader` uses `num_workers=2` over an `IterableDataset`
  that is not sharded across workers. As a result, every packed chunk is yielded twice in a row,
  once by each worker, with independent noise draws. This is what was run for both arms, and it
  is kept unchanged here. 2000 steps therefore process 24,000 chunks of at most 4096 tokens
  (at most 98 M token positions), made up of 12,000 distinct chunks. Documents are not packed, so
  each chunk is one document here: replaying the data stream of the reported runs gives 12,000
  Nemotron-SFT-Code documents with 3.55 M tokens in total (296 tokens on average, none longer than
  4096), i.e. 7.1 M token positions processed. Rank r reads the shards `files[r::4]` of the
  seed-42 shuffle in turn, from their first row, so all 12,000 documents are the first 3,000 rows
  of four shards (`part_000047`, `part_000048`, `part_000052`, `part_000008` for ranks 0–3). The
  200-step runs read the first 300 rows of the same shards: 1,200 documents, 0.37 M tokens.
- **Determinism.** Launching again with the same world size reproduced the step-1 MDLM loss
  exactly (0.5893).

## License and attribution

This directory is released under the repository's MIT license. `PackedParquetStream` in
`train_llada_mixture.py` re-implements the packing of VeOmni's `process_pretrain_example`
(`veomni/data/data_transform.py`, Copyright 2025 Bytedance Ltd. and/or its affiliates,
Apache-2.0). The corruption process follows the mixture collator of the 0.5B training code.
LLaDA-8B-Base and Nemotron-Pretraining-SFT-v1 are subject to their own licences.
