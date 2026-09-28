# LLaDA-8B-Base LoRA corrective training

This directory contains the code for the 8B transfer experiment: a controlled, compute-matched
comparison between absorbing-only masked-diffusion fine-tuning (**MDLM**) and the corrective
mixture objective (**CDLM**) on `GSAI-ML/LLaDA-8B-Base`. The two arms use the same corpus, steps,
schedule, seed, batch and LoRA configuration. Only `mixture_prob` and `noise_token_wt` differ.

## Results reproduced

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
truncates each completion at the first "```" and applies the same rule to every arm. It leaves
every base completion unchanged (checked on all 24 base cells). Both numbers are reported above. At T=4
(`--refined_steps 5`) the de-fenced means are 59.5 / 60.9 / 62.3 (base / MDLM / CDLM) and the raw
means are 59.5 / 46.0 / 51.8.

## Files

| file | purpose |
|---|---|
| `train_llada_mixture.py` | Standalone trainer: HF `AutoModel` + `peft` LoRA + torchrun DDP. The corruption process is `forward_process` (L111) and the loss is `compute_loss` (L170). |
| `run_train.sh` | Launches one arm with the paper's hyperparameters. |
| `refine_code_lora.py` | Wraps the repo-level `refine_code.py`. It loads a LoRA adapter, merges it into the bf16 base and then runs the unchanged CRB refinement. |
| `run_crb_cell.sh` | Runs one CRB cell (arm × dataset × error type × n_replace × steps) on one GPU: refinement, then raw and de-fenced evaluation with the repo-level `evaluate_code.py`. |
| `run_crb_parallel.sh` | Runs the full 72-cell sweep (3 arms × 4 datasets × 3 error types × {2, 5} steps) with one queue per GPU. |
| `defence.py` | The markdown-fence stripping rule (see above). |
| `aggregate_crb.py` | Builds the raw and de-fenced Pass@1 tables from the sweep outputs. |
| `eval_confidence.py`, `run_conf_eval.sh` | Measure the confidence gap and Top-K localisation hit rate (K = 1..6). |
| `requirements.txt` | The package versions used for the experiment. |

## Training objective

Both arms train on the LLaDA absorbing loss. For each sequence the trainer draws
`t ~ U(0,1)` and masks every token independently with probability `t`. The loss is
`L_absorb = Σ_masked CE / t / (B·L)`. LLaDA is bidirectional, so no logits are shifted.

CDLM also corrupts visible tokens. After masking, each remaining visible token is replaced with
probability `mixture_prob` by a uniformly sampled token that is neither the mask token nor the
original token. Those positions are supervised towards the original token:
`L = L_absorb + noise_token_wt · mean_{replaced} CE`. Clean visible tokens are never supervised
(`clean_token_wt = 0`). Absorbing noise and mixture noise come from separate RNG streams, so the
masking pattern is identical in both arms.

## Hyperparameters (both arms)

| knob | value |
|---|---|
| base model | `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` (8.18 B params, mask id 126336) |
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
launcher's default `DATA_DIR`. The G6 runs read a copy downloaded at Hub revision
`3f1a5b884d0b890f02ead979ce698dc95debd953`; the script pins `af7991c59eeb5e53a98bb6b1ee7a96cc3754eb39`,
and the Hub lists the same 78 files (`part_000000.parquet` to `part_000077.parquet`) with identical
names and sizes at both revisions. Point `DATA_DIR` elsewhere if you stored the data somewhere else. The trainer uses every
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
and per-rank RNG streams as the paper. To change the model path or any other flag, call the
trainer directly:

```bash
torchrun --nproc_per_node=4 training/llada8b_lora/train_llada_mixture.py --arm cdlm \
    --model_path GSAI-ML/LLaDA-8B-Base --data_dir <DATA_DIR> --output_dir <OUT> \
    --max_steps 2000 --cosine_horizon 20000 --warmup_ratio 0.01 --lr 3e-4 \
    --weight_decay 0.01 --max_grad_norm 1.0 --micro_batch_size 1 --global_batch_size 12 \
    --max_seq_len 4096 --seed 42 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --clean_token_wt 0.0 --save_steps 0 --log_every 20
```

## Evaluation on CRB

### CRB inputs

The experiment reads the LLaDA-tokenised CRB files generated with `--data_num 2`. They go under
`buggy_datasets/` at the repo root:

```
buggy_datasets/<ds>/LLaDA-8B-Base_<err>_2_wrong_1.jsonl                       # localisation
buggy_datasets/<ds>/evaluated/LLaDA-8B-Base_<err>_2_wrong_1_evaluated.jsonl   # refinement
```

Here `<ds>` is one of `human-eval`, `human-eval+`, `mbpp`, `mbpp+` and `<err>` is one of
`operator`, `var`, `literal`. To create them with the repo's standard pipeline (see
`examples/test_human-eval_llada.sh`):

```bash
python codecorrection/generate.py --dataset <ds> --error_type <err> --n_replace 1 --data_num 2 \
    --model_name GSAI-ML/LLaDA-8B-Base --data_path buggy_datasets --deduplicate
python evaluate_code.py --results_file buggy_datasets/<ds>/LLaDA-8B-Base_<err>_2_wrong_1.jsonl \
    --output_file buggy_datasets/<ds>/evaluated/LLaDA-8B-Base_<err>_2_wrong_1_evaluated.jsonl \
    --dataset <ds> --map_prompt2completion --no_postprocess
```

With the default seed 42, this regenerates the `operator` files byte-for-byte. The `var` and
`literal` files cannot be regenerated exactly. Their replacement candidates come from a Python
`set` of strings (`codecorrection/generate.py`, `all_elements_text`), so the result depends on
`PYTHONHASHSEED` and changes from run to run. To reproduce the reported numbers exactly, use the
released CRB files rather than regenerating them.

### Localisation (confidence gap, Top-K hit rate)

```bash
bash training/llada8b_lora/run_conf_eval.sh outputs/llada8b_lora/full_conf \
    outputs/llada8b_lora/full_mdlm/final outputs/llada8b_lora/full_cdlm/final
```

This evaluates base, MDLM and CDLM on HumanEval with `n_replace=1` and all three error types. It
writes `conf_{base,mdlm,cdlm}.json` with per-file and pooled `gather_gap` and `gather_hit@K`.
Any extra arguments are passed on to `eval_confidence.py`, for example
`--datasets human-eval mbpp` or `--n_replace 1 3`. The `max_*` fields use the
max-over-vocabulary confidence instead, which gives a different ranking (see the paper's appendix).

### Pass@1 under iterative refinement

```bash
# full sweep: 3 arms x 4 datasets x 3 error types x refined_steps {2, 5}, n_replace=1
GPUS="0 1 2 3" bash training/llada8b_lora/run_crb_parallel.sh 1
python training/llada8b_lora/aggregate_crb.py --n_replace 1 --out outputs/llada8b_lora/pass1_tables.md
```

The sweep reads its adapters from `MDLM_ADAPTER` and `CDLM_ADAPTER`, which default to
`${OUTPUT_DIR}/full_{mdlm,cdlm}/final`. Per-GPU logs go to `${LOG_DIR}/crb_gpu<N>.log`. Refined
outputs go to `g6v_{base,mdlm,cdlm}_results/` and per-step histories go to
`g6v_*_history/`, both under the repo root. The histories can be large; you can delete them once
Pass@1 has been computed.

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

## Reproducibility notes

- **Evaluation pipeline.** During the experiment, refinement ran against a frozen snapshot of the
  CRB pipeline. That snapshot's `evaluate_code.py` and `sanitize.py` are byte-identical to the
  repo-level files. Its `refine_code.py`, `llada_sample.py` and `utils.py` add batched padding,
  other baselines and a different split of samples across ranks. None of these changes affects
  LLaDA at `--batch_size 1`. A CPU check compared the repo-level pipeline with the snapshot on
  real CRB inputs, using the real LLaDA tokenizer and a small deterministic stand-in for the
  model. Collation, sampling trajectory, decoded text and history were identical in all 144
  test cases (72 samples × `--refined_steps` 2 and 5).
- **Batch size.** Batched bf16 inference is not bit-identical to `--batch_size 1`. Keep
  `--batch_size 1` for every arm. Sharding cells across GPUs does not change any result.
- **Evaluation noise.** Test execution sometimes times out. Identical inputs have given results
  0.4 points apart on one cell, so treat differences below about 0.5 points as noise.
- **Data loader.** The trainer's `DataLoader` uses `num_workers=2` over an `IterableDataset`
  that is not sharded across workers. As a result, every packed chunk is yielded twice in a row,
  once by each worker, with independent noise draws. This is what was run for both arms, and it
  is kept unchanged here. 2000 steps therefore process 24,000 chunks of at most 4096 tokens
  (at most 98 M token positions), made up of 12,000 distinct chunks.
- **Determinism.** Launching again with the same world size reproduced the step-1 MDLM loss
  exactly (0.5893).

## License and attribution

This directory is released under the repository's MIT license. `PackedParquetStream` in
`train_llada_mixture.py` re-implements the packing of VeOmni's `process_pretrain_example`
(`veomni/data/data_transform.py`, Copyright 2025 Bytedance Ltd. and/or its affiliates,
Apache-2.0). The corruption process follows the mixture collator of the 0.5B training code.
LLaDA-8B-Base and Nemotron-Pretraining-SFT-v1 are subject to their own licences.
