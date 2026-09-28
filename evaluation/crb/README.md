# CRB evaluation

`run_crb.sh` scores one Open-dCoder-family model on the Code Revision Benchmark (CRB):
error localisation from the model's own confidences, and correction Pass@1 after T = 1..4
rounds of confidence-based remasking. It runs the repository's own pipeline
(`refine_code.py`, `llada_sample.py`, `utils.py`, `evaluate_code.py`, `sanitize.py`) with
`PYTHONPATH=<repo>/Open-dLLM`.

```bash
# from the repository root, in an environment with torch, transformers, datasets, evaluate,
# huggingface_hub and numpy (the paper's numbers were produced with torch 2.5.0,
# transformers 4.54.1, datasets 3.6.0, evaluate 0.4.5 on an A100)
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 cdlm --gpus 0
bash evaluation/crb/run_crb.sh /path/to/hf_checkpoint my_model --gpus 0,1 --jobs 4
bash evaluation/crb/run_crb.sh fredzzp/open-dcoder-0.5B base --nr 1          # n_replace=1 only
```

Run `bash evaluation/crb/run_crb.sh` without arguments for all options (`--nr`, `--steps`,
`--datasets`, `--error_types`, `--out_dir`, `--jobs`, `--port_base`, `--eval_jobs`, `--phase`,
`--purge`). Outputs go to `evaluation/crb/outputs/` by default:

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

On one A100 the full grid took about 35 min of refinement (`--jobs 3`, a few GB of GPU
memory per job) and 45 min of evaluation (`--eval_jobs 8`) on a lightly loaded machine;
`--nr 1` is about a fifth of the work. Evaluation executes every program on the CPU and
slows down on a busy machine.

Refinement is deterministic: rerunning the CDLM checkpoint reproduced every completion and
every step-0 confidence bitwise. The only nondeterminism is the 8 s execution timeout: a
program close to it (e.g. HumanEval+ `HumanEval/139`, about 9 s) can pass or time out
depending on machine load; `n_timeout_total` in the summary counts such failures. Keep
`--eval_jobs` moderate on a busy machine.

## Inputs

By default the paper's input set (60 files, `<tag>_<error>_2_wrong_<n>_evaluated.jsonl`
with tag `Open-Dcoder-0.5B-mixture-mdm-step2000`, tokenised with the Open-dCoder tokenizer)
is downloaded from the Hugging Face dataset `Shuibai12138/crb-paper-inputs`
(directory `open-dcoder-0.5B/`) at `CRB_INPUTS_REVISION` (default `main`), and checked
against `paper_inputs.md5`. `CRB_INPUTS_DIR=<dir>` uses a local copy instead
(`<dir>/open-dcoder-0.5B/<dataset>/evaluated/`, `<dir>/buggy_datasets/<dataset>/evaluated/` or
`<dir>/<dataset>/evaluated/`). `CRB_OFFLINE=1` never contacts the Hub.

The evaluation datasets are cached at pinned revisions (`crb_evaldata.py`) and all
evaluation runs offline; the run stops if the test fields differ from the pinned ones
(`--allow_data_drift` to continue anyway).

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
