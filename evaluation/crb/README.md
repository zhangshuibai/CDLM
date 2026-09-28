# CRB evaluation

`run_crb.sh` scores one Open-dCoder-family model on the Code Revision Benchmark (CRB):
error localisation from the model's own confidences, and correction Pass@1 after T = 1..4
rounds of confidence-based remasking. It runs the repository's own pipeline
(`refine_code.py`, `llada_sample.py`, `utils.py`, `evaluate_code.py`, `sanitize.py`) with
`PYTHONPATH=<repo>/Open-dLLM`.

```bash
# from the repository root, in the evaluation environment of evaluation/ENVIRONMENT.md
# (liger-kernel included: without it the numerics change, and the launcher stops)
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 cdlm --gpus 0 \
    --model_revision 5a7170e7c2333e41d5312c690ac4818722a76c3a
bash evaluation/crb/run_crb.sh /path/to/hf_checkpoint my_model --gpus 0,1 --jobs 4
bash evaluation/crb/run_crb.sh fredzzp/open-dcoder-0.5B base --nr 1 \
    --model_revision d0d86d5b99960c05258bb1f8265dd91564dbac67              # n_replace=1 only
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct cdlm_oci --nr 1 \
    --model_revision 8eb87fe2ab6850ced7606c8a678c84f1b28fd170 --phase preflight   # checks only, CPU
```

`bash evaluation/crb/run_crb.sh --help` prints every option:

| option | default | meaning |
|---|---|---|
| `--gpus` | `0` | comma-separated GPU ids |
| `--jobs` | `3` | concurrent refinement processes, one GPU each; job slot i runs on GPU i mod (number of `--gpus`), so several jobs can share a GPU |
| `--port_base` | `29500` | rendezvous ports `port_base` .. `port_base+jobs-1` |
| `--eval_jobs` | `8` | concurrent CPU scoring processes |
| `--nr` | `"1 2 3 4 5"` | n_replace levels |
| `--steps` | `"2 3 4 5"` | `--refined_steps` values; T = steps - 1 |
| `--datasets` | `"human-eval human-eval+ mbpp mbpp+"` | |
| `--error_types` | `"operator var literal"` | |
| `--out_dir` | `evaluation/crb/outputs` | |
| `--model_revision` | none | Hub revision of MODEL; the snapshot is then used like a local directory |
| `--phase` | `all` | `all`, `refine`, `eval`, `metrics`, or `preflight` (only the checks, on the CPU; exit 0 if all pass) |
| `--purge` | off | delete histories and refined jsonl once the summary is complete |
| `--allow_data_drift` | off | continue if the evaluation datasets differ from the pinned ones |

Environment variables: `CRB_PYTHON`, `CRB_INPUTS_DIR`, `CRB_INPUTS_REPO`, `CRB_INPUTS_REVISION`,
`CRB_OFFLINE`, `HF_SCRIPTS_VERSION` (default `v0.4.0`). Outputs go to `evaluation/crb/outputs/` by
default:

| file | content |
|---|---|
| `results/<label>_summary.json` | `headline_nr1_macro12` (the paper's numbers), macros per n_replace and T, per-dataset macros, every cell, run metadata |
| `results/<label>_per_cell.csv` | one row per (dataset, error type, n_replace, T) |
| `results/<label>_per_sample.csv` | `test_passed` of every refined program |
| `runs/<label>_{results,history}/` | refined and evaluated jsonl, per-step histories (`--purge` deletes histories and refined jsonl once the summary is complete) |

## Protocol

The grid is 4 datasets (HumanEval, HumanEval+, MBPP, MBPP+) x 3 error types (operator,
var, literal) x n_replace 1..5 x T = 1..4. Refinement uses `--refine_setting remove_all`,
`--algorithm self_conf-remask:vanilla`, remasking a token iff its confidence is <= 0.9,
temperature 0.0 and batch size 1, with `--refined_steps` 2..5 (T = steps - 1: the last step
never remasks). Every job is a single process on one GPU. The refined programs are executed
by `evaluate_code.py --no_postprocess` with its default 8 s timeout.

* Confidence c_i is p(z_i | z), the probability of the currently visible token at body
  position i in the first forward pass over the buggy program.
* Confidence gap: mean c over clean positions minus mean c over corrupted positions.
  Top-K: a corrupted position is among the K lowest-confidence positions.
* Pass@1: fraction of the test-failing corrupted programs that pass after refinement.
* Every metric is computed per (dataset, error type) cell; the headline is the unweighted
  mean over the 12 cells at n_replace = 1.

Published models on the paper's input set (headline, n_replace = 1):

| model | T=1 | T=2 | T=3 | T=4 | gap | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|---|---|---|
| `fredzzp/open-dcoder-0.5B` | 0.1452 | 0.2404 | 0.2549 | 0.2632 | 0.1047 | 0.1472 | 0.3447 | 0.5112 |
| `Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000` (MDLM) | 0.1381 | 0.2363 | 0.2463 | 0.2453 | 0.0976 | 0.1470 | 0.3427 | 0.5028 |
| `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000` (CDLM) | 0.1883 | 0.2789 | 0.2814 | 0.2853 | 0.1636 | 0.2277 | 0.4499 | 0.6058 |
| `Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct` (CDLM-OCI, not a paper model) | 0.2196 | 0.3018 | 0.3029 | 0.3070 | 0.1919 | 0.2517 | 0.4883 | 0.6495 |
| `Shuibai12138/Open-Dcoder-0.5B-MDLM-OpenCodeInstruct` (MDLM-OCI, not a paper model) | 0.1401 | 0.2226 | 0.2352 | 0.2387 | 0.0987 | 0.1445 | 0.3575 | 0.5199 |

The pinned revisions, weight hashes and commands for all five are in
[`evaluation/README.md`](../README.md#models). The two OpenCodeInstruct checkpoints are reference
checkpoints for the released training path; no paper number comes from them.

On one A100 the full grid took about 35 min of refinement (`--jobs 3`, a few GB of GPU
memory per job) and 45 min of evaluation (`--eval_jobs 8`) on a lightly loaded machine;
`--nr 1` is about a third of the work (12,852 of 37,028 programs). Evaluation executes every
program on the CPU and slows down on a busy machine.

Refinement is deterministic: rerunning the CDLM checkpoint reproduced every completion and
every step-0 confidence bitwise. The only nondeterminism is the 8 s execution timeout: a
program close to it (e.g. HumanEval+ `HumanEval/139`, about 9 s) can pass or time out
depending on machine load; `n_timeout_total` in the summary counts such failures. Keep
`--eval_jobs` moderate on a busy machine.

## Inputs

By default the paper's input set (60 files, `<tag>_<error>_2_wrong_<n>_evaluated.jsonl`
with tag `Open-Dcoder-0.5B-mixture-mdm-step2000`, tokenised with the Open-dCoder tokenizer)
is downloaded from the Hugging Face dataset `Shuibai12138/crb-paper-inputs`
(directory `open-dcoder-0.5B/`) at `CRB_INPUTS_REVISION` (default: the pinned upload
`21cae17423b073b152e997746876d6b828b18358`), and checked against `paper_inputs.md5`.
`CRB_INPUTS_DIR=<dir>` uses a local copy instead
(`<dir>/open-dcoder-0.5B/<dataset>/evaluated/`, `<dir>/buggy_datasets/<dataset>/evaluated/` or
`<dir>/<dataset>/evaluated/`). `CRB_OFFLINE=1` never contacts the Hub. A local copy of the
pinned upload:

```bash
hf download Shuibai12138/crb-paper-inputs --repo-type dataset \
    --revision 21cae17423b073b152e997746876d6b828b18358 \
    --include "open-dcoder-0.5B/*" --local-dir crb-paper-inputs
CRB_INPUTS_DIR=crb-paper-inputs bash evaluation/crb/run_crb.sh <MODEL> <LABEL>
```

The evaluation datasets are cached at pinned revisions (`crb_evaldata.py`) and all
evaluation runs offline; the run stops if the test fields differ from the pinned ones
(`--allow_data_drift` to continue anyway). The `code_eval` metric is fetched at
`HF_SCRIPTS_VERSION` (default `v0.4.0`), and the run stops if its files differ from the
verified ones; there is no override.

## Safety checks

* `utils.py` selects the Open-dCoder code path (bidirectional attention, logit shift, mask
  id 151665) only when the model name contains `open-dcoder`, and switches to the LLaDA mask
  id when it contains `llada`. Local directories are therefore symlinked to
  `<out_dir>/models/open-dcoder-0.5B-<label>`; Hub ids without `open-dcoder` are refused.
  `Shuibai12138/CDLM-0.5B` holds the same weights as
  `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000`; use the latter name.
* `config.json` must have `model_type: qwen2`.
* `crb_tokcheck.py` tokenises every input program with the model's tokenizer; the hash must
  equal the one of the tokenizer that generated the inputs, otherwise the stored error
  positions (token indices) would be misaligned.
* The pipeline files and `Open-dLLM/veomni` are hashed; a difference from the release that
  was verified against the paper's numbers prints a warning and is recorded in the metadata.
* `liger-kernel` must be importable (veomni's Qwen2 otherwise runs plain PyTorch layers, which
  changes the numerics). A version of torch, transformers, tokenizers, liger-kernel, triton,
  accelerate, datasets, evaluate or huggingface-hub other than the verified one prints a
  warning; the versions are recorded in `run_meta.json`.

## Building a new input set

```bash
bash evaluation/crb/build_crb_inputs.sh --out_dir crb_inputs_new --tokenizer fredzzp/open-dcoder-0.5B \
    --seed 42 --nr "1 2 3 4 5" --data_num 2
CRB_INPUTS_DIR=crb_inputs_new bash evaluation/crb/run_crb.sh <MODEL> <LABEL>
```

It runs `codecorrection/generate.py --deduplicate` for every cell, then
`evaluate_code.py --map_prompt2completion --no_postprocess` to mark the corrupted programs
that still pass their tests (refinement only uses the failing ones), and writes `INPUTS.md5`
and `crb_inputs_meta.json` (tag, settings, tokenizer hash), which `run_crb.sh` checks. With
the default settings the operator files are byte-identical to the paper's. The paper's var
and literal files cannot be regenerated byte for byte, because the generator used before
the release iterated a Python set, which made its choices depend on `PYTHONHASHSEED`.

## Comparing runs

```bash
python evaluation/crb/crb_table.py --results_dir evaluation/crb/outputs/results \
    --labels base mdlm cdlm --pairs cdlm:mdlm --nr 1          # table + exact-paired McNemar
python evaluation/crb/crb_compare.py --new evaluation/crb/outputs/runs cdlm \
    --ref <other_out_dir>/runs cdlm --nr 1                    # per-sample diff of two runs
```

`crb_table.py` takes any labels, for example `--labels cdlm_oci mdlm_oci --pairs cdlm_oci:mdlm_oci`.
`crb_compare.py` compares, per cell, the refined jsonl and the per-step histories. A run made
with `--purge` has neither; for such cells it compares the `*_results_refined_evaluated.jsonl`
files instead (task id, prompt, completion, step count, `test_passed`), without step-0
confidences or histories, and counts them in `cells_from_evaluated_only`. It exits with
status 1 if it compared no cell (wrong root, prefix or grid options).
