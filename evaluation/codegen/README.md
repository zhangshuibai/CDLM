# Code-generation evaluation (Table 5 top, Appendix Table 9)

`run_codegen_eval.sh` evaluates one Open-dCoder checkpoint on HumanEval, HumanEval+, MBPP and MBPP+
with the protocol of the paper. It runs the repository's own evaluation code
(`Open-dLLM/eval/eval_completion/eval.py`, the vendored `Open-dLLM/lm-evaluation-harness` and
`Open-dLLM/veomni`) and re-implements nothing of the generation or scoring path. For the published
models it reproduces the paper's recorded samples token for token (see [Verification](#verification)).

## Protocol

| | |
| --- | --- |
| Benchmarks | lm-eval tasks `humaneval` (164 problems), `humaneval_plus` (164), `mbpp` (500), `mbpp_plus` (378), zero-shot |
| Samples | n = 10 per problem (`repeats: 10` in the task yaml) |
| Decoding | 128 new tokens in 128 steps, temperature 0.8, top-k 200 (`eval.py` default); `top_p=0.95` is passed as in `run_eval.sh` but `CustomCoder` ignores it |
| Decoders | vanilla = `alg=p2_upgraded`, ReMDM = `alg=p2_upgraded_ReMDM` |
| Metric | unbiased pass@1 and pass@10 of the HF `code_eval` metric, averaged over problems |
| Batch size | 5 for HumanEval(+), 1 for MBPP(+), as in `run_eval.sh` |
| Seeds | lm-eval defaults: `random` 0, `numpy` 1234, `torch` 1234, few-shot 1234 |
| Sharding | 4 virtual ranks: rank r gets problems r, r+4, r+8, ... |

The paper's numbers come from `accelerate launch --num_processes 4 eval.py` (`run_eval.sh`). Each of
the 4 processes seeded its RNGs, took every fourth problem and generated them in order, so the sampled
tokens depend on the rank a problem fell on and on the batch size. `codegen_eval.py` replays lm-eval's
`simple_evaluate` → `evaluate` flow for each rank as an independent single-GPU worker (same seeds,
sharding, `repeats` cloning, padding of short ranks, filters and `process_results`) and merges the 4
shards. The result therefore does not depend on how many GPUs you use, but changing the number of
virtual ranks or the batch size changes the samples.

## Environment

Use the evaluation environment of [`evaluation/requirements.txt`](../requirements.txt) (Python 3.11;
the install order is in its header). The verification below ran in an environment with exactly those
versions (torch 2.5.0+cu121, transformers 4.54.1, datasets 3.6.0, evaluate 0.4.5, liger-kernel 0.5.8,
...) on NVIDIA A100-PCIE-40GB GPUs (driver 560.35.03). liger-kernel matters: when it is importable,
veomni's Qwen2 runs RMSNorm, the MLP and the rotary embedding with its triton kernels, so the sampled
tokens change without it.

Do not pip-install Open-dLLM or its lm-evaluation-harness. The launcher puts the repository's
`Open-dLLM/eval/eval_completion`, `Open-dLLM/lm-evaluation-harness` and `Open-dLLM` on `PYTHONPATH`;
any `lm_eval` or `veomni` installed in the environment is shadowed, and the workers check that both are
imported from the repository.

## Usage

```bash
# one published model, all four benchmarks, both decoders, on GPUs 0-3 with 4 workers per GPU
bash evaluation/codegen/run_codegen_eval.sh \
    Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000@4581e1d4215a4055ccce6eaaf898f27276a8759c \
    outputs/codegen/cdlm 0,1,2,3

# a local HF checkpoint, HumanEval only, vanilla decoder, one GPU
bash evaluation/codegen/run_codegen_eval.sh path/to/checkpoint outputs/codegen/mine 0 4 vanilla humaneval

# spot check: only the first 3 problems of every virtual rank (a few minutes on one GPU)
MAX_DOCS=3 bash evaluation/codegen/run_codegen_eval.sh <model> outputs/codegen/spot 0 4 vanilla humaneval
```

Arguments: `MODEL OUT_DIR [GPUS=0] [PROCS_PER_GPU=4] [ALGS=vanilla,remdm] [TASKS=humaneval,humaneval_plus,mbpp,mbpp_plus]`.
They can also be given as environment variables (`GPUS`, `PROCS_PER_GPU`, `ALGS`, `TASKS`); `PYTHON`
selects the interpreter and `MAX_DOCS` runs a spot check. `MODEL` is a local HF checkpoint directory,
a Hub id, or `hub_id@revision`, which is downloaded at that revision (a Hub id without a revision is
resolved to the current commit, which is recorded in `summary.json` with a warning). Rerunning an
interrupted run with the same `OUT_DIR` skips the shards that finished.

The 4 virtual ranks × decoders × benchmarks (up to 32 shards) run as independent single-GPU worker
processes, `len(GPUS) × PROCS_PER_GPU` at a time; a 0.5B HumanEval worker (batch size 5) uses about
6 GB of GPU memory. Downloads (model, datasets, the `code_eval` metric module) happen once in the
preflight; the workers run with `HF_HUB_OFFLINE=1` and read the local caches only.

**The benchmarks execute model-generated code** (`HF_ALLOW_CODE_EVAL=1`, in subprocesses with the
`code_eval` reliability guard). Run the evaluation in a container or another isolated environment.

### Published models

| Key | Model | Hub id | Revision used by the paper runs | `model.safetensors` sha256 |
| --- | --- | --- | --- | --- |
| `base` | Open-dCoder-0.5B | `fredzzp/open-dcoder-0.5B` | `d0d86d5b99960c05258bb1f8265dd91564dbac67` | `59c1a005…8f77b6e` |
| `mdlm` | MDLM-0.5B | `Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000` | `b78b055f3a1f0e683d3893783c10fbfd939b0cf7` | `4f1f412c…344ff36e` |
| `cdlm` | CDLM-0.5B | `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000` | `4581e1d4215a4055ccce6eaaf898f27276a8759c` | `e43f9fa6…fdf8a680` |

Later commits of the two Shuibai12138 repositories add only a model card; the weights and tokenizer
files are the same. The full hashes are in `reference/recorded_runs.json`.

## What the run checks

Before any GPU work, `run` stops if one of these fails:

- **Code.** Every file in `MANIFEST.md5` (188 files: `eval_completion`, the lm-eval package with only its
  `humaneval/` and `mbpp/` tasks, and `veomni`) must have the listed md5. The list is identical to the
  verified protocol that reproduced the recorded runs. The `code_eval` metric module that the task
  utils load from the Hub (`evaluate.load("code_eval")`) must match the two `hf-evaluate:` entries.
  Only the `humaneval/` and `mbpp/` task directories are indexed, and each worker checks at the end that
  every module it imported from `Open-dLLM/` is in the manifest.
- **Inputs.** Each dataset is loaded at the Hub revision in `inputs/INPUTS.json`, and the per-problem
  document and prompt hashes that lm-eval logs must reproduce the pinned digests. These inputs match the
  recorded samples files of all three published models. `python evaluation/codegen/codegen_eval.py inputs
  [--export DIR]` runs this check on the CPU and can export the problems (`doc_id`, `prompt`, `target`,
  `doc`) as JSON lines.
- **Model.** `config.json` must declare `Qwen2ForCausalLM`, `torch_dtype` bfloat16 and vocabulary size
  151936, the weights must be a single `model.safetensors`, and the tokenizer must define the mask token
  `<M>` with id 151665 and `<|endoftext|>` as EOS and padding token. Without `<M>`, `eval.py` would add
  a new `[MASK]` token and resize the embeddings, and the evaluation would silently run a different model.

If the weights are one of the published models (matched by sha256), the run also compares every
problem with the paper's recorded run: the digest of its 10 raw completions and its number of passing
samples (`reference_check` in `summary.json`; a warning is printed if anything differs).

## Outputs

`OUT_DIR/` contains

- `summary.json`: pass@1 and pass@10 with standard errors per decoder and benchmark, the averages over
  HumanEval / HumanEval+ (`he_avg_*`), MBPP / MBPP+ (`mbpp_avg_*`) and all four (`macro4_*`), the model
  provenance and preflight results, the manifest and input digests, package versions, the repository
  commit and `reference_check`;
- `samples_<task>__<alg>.jsonl`: per problem the raw and sanitized completions, pass@1, pass@10 and the
  lm-eval doc / prompt / target hashes (the fields of lm-eval's `--log_samples` files);
- `metrics_<task>__<alg>.json` and `shards/*.log`, `shards/*.meta.json` (timings, per worker).

Samples files of this script and recorded lm-eval `samples_<task>_*.jsonl` files can be compared:

```bash
# token-level comparison with a recorded lm-eval samples file, or with the paper's run (built-in digests)
python evaluation/codegen/codegen_eval.py compare --mine OUT/samples_humaneval__p2_upgraded.jsonl --recorded samples_humaneval_2025-11-29T17-03-32.375387.jsonl
python evaluation/codegen/codegen_eval.py compare --mine OUT/samples_humaneval__p2_upgraded.jsonl --reference cdlm --task humaneval --alg vanilla

# paired per-problem comparison of two models (same benchmark and decoder): difference of the means,
# paired bootstrap 95% CI and a two-sided sign-flip permutation p-value
python evaluation/codegen/codegen_eval.py paired --a OUT_CDLM/samples_humaneval__p2_upgraded.jsonl --b OUT_BASE/samples_humaneval__p2_upgraded.jsonl
```

## Recorded results

Pass@1 / Pass@10 of the paper's recorded runs (`reference/recorded_runs.json`):

| Model | Decoder | HumanEval | HumanEval+ | MBPP | MBPP+ |
| --- | --- | --- | --- | --- | --- |
| Open-dCoder-0.5B | vanilla | 18.84 / 36.59 | 16.71 / 31.71 | 15.74 / 35.00 | 23.11 / 50.24 |
| Open-dCoder-0.5B | ReMDM | 18.23 / 32.93 | 16.40 / 29.27 | 16.00 / 36.60 | 23.19 / 48.54 |
| MDLM-0.5B | vanilla | 19.02 / 38.41 | 16.95 / 34.15 | 10.62 / 31.20 | 17.71 / 44.91 |
| MDLM-0.5B | ReMDM | 19.82 / 38.41 | 17.20 / 34.15 | 10.84 / 32.80 | 16.97 / 44.42 |
| CDLM-0.5B | vanilla | 21.59 / 41.46 | 20.55 / 39.02 | 13.10 / 35.20 | 20.73 / 47.22 |
| CDLM-0.5B | ReMDM | 21.16 / 42.68 | 20.00 / 37.80 | 14.14 / 35.00 | 21.06 / 47.60 |

MBPP+ has 378 problems, which do not split evenly over 4 ranks. As in lm-eval, the two short ranks are
padded by repeating their last request, so problems 374 and 375 have 20 samples; their pass@k uses all
20, in the recorded runs and here.

## Verification

With the environment above on A100-PCIE-40GB GPUs:

- A frozen copy of the same code (byte-identical to the files in `MANIFEST.md5`) reproduced three full
  recorded runs bitwise (all completions and per-problem pass@k identical): CDLM-0.5B HumanEval vanilla,
  Open-dCoder-0.5B HumanEval vanilla and Open-dCoder-0.5B MBPP+ ReMDM, plus spot checks that cover all
  three models, both decoders and all four benchmarks. Re-scoring the recorded completions of all 24
  runs (3 models × 2 decoders × 4 benchmarks) gives the recorded per-problem pass@1 for every problem.
- This script, running the repository files (one GPU, 2 workers): the full CDLM-0.5B HumanEval vanilla
  run reproduces the recorded run exactly (1,640 of 1,640 completions and the pass@1 / pass@10 of all 164
  problems identical; pass@1 0.21585, pass@10 0.41463; 22 minutes). So do two spot checks: CDLM-0.5B
  HumanEval vanilla (first 3 problems of every rank) and Open-dCoder-0.5B MBPP+ ReMDM (first 2).

`summary.json` reports means over problems in `doc_id` order, while lm-eval sums in rank order, so the
last digits of a mean can differ from the recorded `results_*.json` (for example 0.21585365853658547
against 0.21585365853658545) although every per-problem value is identical.

Other GPUs or library versions can change the sampled tokens. The run is then a different sample of the
same protocol, and `reference_check` reports how many problems differ from the recorded run.

## Runtime

On one A100-PCIE-40GB, HumanEval with one decoder takes about 20 minutes with 2 or 4 workers, and
MBPP+ with ReMDM about 80 minutes with 4 workers (about 18,700 worker-seconds of generation).
MBPP and MBPP+ run at batch size 1 and dominate, and on them ReMDM is about twice as slow as vanilla.
Estimated from these timings, the full grid (four benchmarks, both decoders) takes about 6 hours on
one such GPU, and proportionally less on several.
