# Evaluation

This directory holds the two evaluations of Open-dCoder-0.5B-family checkpoints: the base model,
MDLM-0.5B, CDLM-0.5B, the two OpenCodeInstruct reference checkpoints, or a checkpoint trained with
[`training/`](../training/README.md). Each one has a single launcher. The launcher runs the
repository's own code, checks its code, packages, inputs and model before using a GPU, and writes a
summary. Both launchers can run only these checks, on the CPU (see [Checks only](#checks-only-cpu)).

| | CRB | Code generation |
| --- | --- | --- |
| Measures | Error localisation from the model's confidences, and correction Pass@1 after T = 1..4 rounds of confidence-based remasking, on corrupted HumanEval, HumanEval+, MBPP and MBPP+ programs | From-scratch generation on HumanEval, HumanEval+, MBPP and MBPP+ (Pass@1 and Pass@10, vanilla and ReMDM decoding). It checks that corrective training does not hurt generation |
| Reproduces | The CRB numbers of CDLM-0.5B: Table 4 (CDLM row), Table 3 (CDLM, τ = 0.9) and Table 7 (CDLM column) | Table 5 (top) and Table 9, all three models |
| Launcher | [`crb/run_crb.sh`](crb/run_crb.sh) | [`codegen/run_codegen_eval.sh`](codegen/run_codegen_eval.sh) |
| Details | [`crb/README.md`](crb/README.md) | [`codegen/README.md`](codegen/README.md) |

Table numbers refer to the NeurIPS 2026 version of the paper.

## Environment

Both evaluations use one environment, which is not the training environment. Its install steps,
version pins and Hugging Face revisions are in [`ENVIRONMENT.md`](ENVIRONMENT.md). It needs Linux
x86_64 with glibc 2.28 or newer. Run the commands in this order: `requirements.txt` on its own does
not install (see [ENVIRONMENT.md](ENVIRONMENT.md#install)).

```bash
conda create -n cdlm-eval python=3.11.13 -y && conda activate cdlm-eval
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r evaluation/requirements.txt
pip check
```

Run the launchers from the repository root with this environment active. They set `PYTHONPATH`
themselves. Do not `pip install` `Open-dLLM/` or its `lm-evaluation-harness`, and never put `training/`
on `PYTHONPATH`. `CRB_PYTHON` (CRB) and `PYTHON` (code generation) select another interpreter.

Both launchers set `HF_SCRIPTS_VERSION=v0.4.0` (unless it is already set), so `evaluate` fetches the
verified `code_eval` metric, and they stop if its files differ from the verified ones. They also stop
if `liger-kernel` cannot be imported, and warn if torch, transformers, tokenizers, liger-kernel,
triton, accelerate, datasets, evaluate or huggingface-hub differ from the pinned versions.

A fresh environment built with these commands has the same versions of all 72 packages as the
verified environment. It was checked on the CPU only: the tokenization of the CRB inputs, the scoring
of CRB programs, the lm-eval inputs and the rescoring of recorded samples all match. The GPU results
below come from an environment with the same package versions, not from a freshly built one.

> [!WARNING]
> Both evaluations execute model-generated programs and the benchmark tests on your machine
> (`HF_ALLOW_CODE_EVAL=1`, which the launchers set). `code_eval` runs each program in a child process
> with a time limit, which is not isolation: the program runs as your user, with your files and your
> network. Run the evaluations in a container or VM that holds no credentials and nothing you need to
> keep, and without network access once the models and datasets are cached. See
> [Executing generated code](ENVIRONMENT.md#executing-generated-code).

## Models

| Model | Hub id | Pinned revision | `model.safetensors` sha256 |
| --- | --- | --- | --- |
| Open-dCoder-0.5B (base) | `fredzzp/open-dcoder-0.5B` | `d0d86d5b99960c05258bb1f8265dd91564dbac67` | `59c1a005f4b672bdd3bbdab6258b283dff3b87bab5cc4bbc0d479e8388f77b6e` |
| MDLM-0.5B | `Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000` | `fa5eef962d343a0d813d1f963a5d44c39deaed45` | `4f1f412cdac13553b76fd8de569ae9ed5a95447dfabaaf16da508d09344ff36e` |
| CDLM-0.5B | `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000` | `5a7170e7c2333e41d5312c690ac4818722a76c3a` | `e43f9fa6b4cccfc18a2bac8925d64f5a020fa6a6d34db2c801220e22fdf8a680` |
| CDLM-OCI (not a paper model) | `Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct` | `8eb87fe2ab6850ced7606c8a678c84f1b28fd170` | `3ae362eb296006bcd139234967bcc5503296b7912261fda337b867d44713acd4` |
| MDLM-OCI (not a paper model) | `Shuibai12138/Open-Dcoder-0.5B-MDLM-OpenCodeInstruct` | `535b36930a32109c87986c341bf554cc42e76b2e` | `31521c7d9f5c02fd2d6b4769ae8d99490c4261af0bc5b566d42d0076221067c4` |

- The paper's runs loaded MDLM-0.5B at `b78b055f3a1f0e683d3893783c10fbfd939b0cf7` and CDLM-0.5B at
  `4581e1d4215a4055ccce6eaaf898f27276a8759c`. The pinned revisions add only a model card
  (`README.md`); every other file is identical.
- Pin the revision when you run a launcher: `--model_revision REV` for CRB, `hub_id@REV` for code
  generation. A bare Hub id resolves to whatever `main` is at run time.
- Both launchers record the sha256 of the weights they loaded.

**For CRB, the model name selects the code path.** `utils.py` and `llada_sample.py` route on the
model name:

- If the lower-cased name contains `open-dcoder`, the model is loaded as `veomni`'s Qwen2, with
  bidirectional attention, logits shifted by one position and mask id 151665.
- A name that contains `llada` gets LLaDA's mask id instead. This check runs first.
- Any other name is loaded with `AutoModel` and is not evaluated as an Open-dCoder model.

`run_crb.sh` enforces this rule:

- It passes a Hub id through unchanged and refuses one without `open-dcoder`.
  `Shuibai12138/CDLM-0.5B` holds the CDLM-0.5B weights but is refused for this reason; use
  `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000`.
- It symlinks a local directory to `<out_dir>/models/open-dcoder-0.5B-<LABEL>`, so a local
  directory can have any name.
- `LABEL` must not contain `llada` or `dream`. For a local directory, `--out_dir` must not contain
  `llada` or `dream-v0` either.

If you call `refine_code.py` directly, give it a path that contains `open-dcoder` and not `llada`.

Code generation has no name rule. It loads every checkpoint with `veomni`'s Qwen2 and checks
`config.json` and the tokenizer instead: a bfloat16 `Qwen2ForCausalLM`, a single `model.safetensors`,
and the mask token `<M>` with id 151665.

### OpenCodeInstruct reference checkpoints

CDLM-OCI and MDLM-OCI are public reference checkpoints for the released training path. They are not
the paper's models, and no paper number comes from them.

- They were trained from `fredzzp/open-dcoder-0.5B` on `nvidia/OpenCodeInstruct` (revision
  `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`) with
  `ARM=cdlm bash training/scripts/train_0.5b_opencodeinstruct.sh` and `ARM=mdlm` respectively, at
  commit `5e52812`. CDLM-OCI took about 19 min on 4 A100-PCIE-40GB GPUs.
- Like the paper's 0.5B checkpoints, they were trained with the 0.5B trainer, whose effective gradient
  differs from the documented per-token loss weighting (see [training/README.md](../training/README.md)).
  Evaluation runs forward passes only and is not affected.
- Their weights have not changed since the first upload. Later commits changed only `config.json`
  (they added `torch_dtype`, which transformers older than 4.56 needs) and `README.md`.

Reference CRB numbers (paper input set, n_replace = 1, mean over the 12 dataset × error-type cells,
τ = 0.9):

| Model | Pass@1 T=1 | T=2 | T=3 | T=4 | Confidence gap | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CDLM-OCI | 0.2196 | 0.3018 | 0.3029 | 0.3070 | 0.1919 | 0.2517 | 0.4883 | 0.6495 |
| MDLM-OCI | 0.1401 | 0.2226 | 0.2352 | 0.2387 | 0.0987 | 0.1445 | 0.3575 | 0.5199 |

- These come from the reference driver on the exported checkpoints, which have the same weights.
- `run_crb.sh` on the Hub id of CDLM-OCI reproduced all 96 refined and history files byte for byte,
  and 44 of the 48 Pass@1 cells. Each of the other 4 cells differs by one program, HumanEval/139,
  which runs just over the 8 s timeout (see [Paper numbers](#paper-numbers)). The largest change of a
  headline value was 0.00086.
- Paired exact McNemar test, CDLM-OCI against MDLM-OCI (3,213 programs per T): +0.0756 at T = 1
  (p = 6e-43) and +0.0784 at T = 4 (p = 6e-27).

Reference code-generation numbers: CDLM-OCI on HumanEval with the vanilla decoder gives pass@1 0.2165
and pass@10 0.4146. This run used `run_codegen_eval.sh` with the Hub id, which resolved to
`6a3f74ef71272553948f2ebcb307acc702820f91`. That revision differs from the pinned one only in
`README.md`. A second run at the pinned revision `8eb87fe2ab6850ced7606c8a678c84f1b28fd170`, with the
launcher at the v1.0.1 release, reproduced all 1,640 completions and every per-problem pass@k.
MDLM-OCI was not scored on code generation. [`codegen/reference/recorded_runs.json`](codegen/reference/recorded_runs.json) holds
per-problem digests of recorded runs, and these checkpoints have none. So `reference_check` stays
empty for them, and `codegen_eval.py compare --reference` does not accept them.

The commands:

```bash
# CRB, n_replace = 1 (the 48 cells behind the numbers above)
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct cdlm_oci --gpus 0 --nr 1 \
    --model_revision 8eb87fe2ab6850ced7606c8a678c84f1b28fd170
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-MDLM-OpenCodeInstruct mdlm_oci --gpus 0 --nr 1 \
    --model_revision 535b36930a32109c87986c341bf554cc42e76b2e
python evaluation/crb/crb_table.py --results_dir evaluation/crb/outputs/results \
    --labels cdlm_oci mdlm_oci --pairs cdlm_oci:mdlm_oci --nr 1

# code generation, HumanEval, vanilla decoder
bash evaluation/codegen/run_codegen_eval.sh \
    Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct@8eb87fe2ab6850ced7606c8a678c84f1b28fd170 \
    outputs/codegen/cdlm_oci 0 4 vanilla humaneval
```

## Checks only (CPU)

Both launchers can run only the checks they do before any GPU work, on a machine without a GPU. They
download what is missing, print the weights' sha256 and exit 0 if every check passes:

```bash
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct cdlm_oci --nr 1 \
    --model_revision 8eb87fe2ab6850ced7606c8a678c84f1b28fd170 --phase preflight
PREFLIGHT_ONLY=1 bash evaluation/codegen/run_codegen_eval.sh \
    Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct@8eb87fe2ab6850ced7606c8a678c84f1b28fd170 \
    outputs/codegen/cdlm_oci 0 4 vanilla humaneval
```

- **CRB** checks the pipeline files, the package versions, the input files (md5), the model's config,
  its tokenizer on every input program, the evaluation datasets and the `code_eval` module. It writes
  only logs and symlinks under `--out_dir`, and not the run metadata.
- **Code generation** checks the 188 manifest files, the package versions, the model's config and
  tokenizer, and the inputs and `code_eval` module of each task.
  `python evaluation/codegen/codegen_eval.py preflight --model M [--algs A] [--tasks T]` runs the same
  checks.

## CRB

### Running it

```bash
# a Hub id at its pinned revision: all 240 cells (4 datasets x 3 error types x n_replace 1..5 x T 1..4)
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 cdlm --gpus 0 \
    --model_revision 5a7170e7c2333e41d5312c690ac4818722a76c3a

# only n_replace = 1 (the 48 cells behind the headline numbers)
bash evaluation/crb/run_crb.sh Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 cdlm --gpus 0 --nr 1 \
    --model_revision 5a7170e7c2333e41d5312c690ac4818722a76c3a

# a local HF checkpoint (safetensors weights, config.json with model_type qwen2, tokenizer files)
bash evaluation/crb/run_crb.sh /path/to/checkpoint my_model --gpus 0,1 --jobs 4
```

- `--model_revision REV` pins the Hub revision. The snapshot is then symlinked like a local
  directory.
- `--jobs` sets the number of concurrent refinement processes (default 3). Each is a single-GPU
  process that needs a few GB. Job slot i runs on GPU i mod n of the n GPUs in `--gpus`, so with
  `--gpus 0 --jobs 3` all three share GPU 0.
- `--eval_jobs` sets the number of CPU scoring processes (default 8).
- `--phase preflight` runs only the checks (see [Checks only](#checks-only-cpu)).
- `--purge` deletes the per-step histories and refined files once the summary is complete.
- `bash evaluation/crb/run_crb.sh --help` lists every option.

The protocol:

- Refinement: `--refine_setting remove_all`, `--algorithm self_conf-remask:vanilla`. A token is
  remasked if its confidence is at most τ = 0.9. Temperature 0, batch size 1.
- `--refined_steps` 2..5, which gives T = 1..4. The last step never remasks.
- The confidence of a token is the probability of the visible token, p(z_i | z).
- The headline is the unweighted mean over the 12 dataset × error-type cells at n_replace = 1.
- Batch size is fixed at 1. With larger batches, `refine_code.py` would let padding into the
  Open-dCoder models' bidirectional attention.

### Inputs

By default the launcher downloads the paper's input set: the directory `open-dcoder-0.5B/` of the
Hugging Face dataset `Shuibai12138/crb-paper-inputs`, at `CRB_INPUTS_REVISION` (default: the pinned
upload `21cae17423b073b152e997746876d6b828b18358`).

- The set has 60 evaluated files (4 datasets × 3 error types × n_replace 1..5), with 9,778
  corrupted programs. The 9,257 that fail their tests are refined and scored.
- The files were generated with the Open-dCoder tokenizer. All Open-dCoder-0.5B-derived checkpoints
  share it, so every such model is scored on the same programs.
- The launcher checks the files against [`crb/paper_inputs.md5`](crb/paper_inputs.md5). It also
  checks that the model's tokenizer reproduces the token ids that the stored error positions refer
  to.

To use a local copy:

```bash
hf download Shuibai12138/crb-paper-inputs --repo-type dataset \
    --revision 21cae17423b073b152e997746876d6b828b18358 \
    --include "open-dcoder-0.5B/*" --local-dir crb-paper-inputs
CRB_INPUTS_DIR=crb-paper-inputs bash evaluation/crb/run_crb.sh <MODEL> <LABEL> --gpus 0
```

`hf` comes with huggingface-hub 0.34.4. The same download in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("Shuibai12138/crb-paper-inputs", repo_type="dataset",
                  revision="21cae17423b073b152e997746876d6b828b18358",
                  allow_patterns=["open-dcoder-0.5B/**"], local_dir="crb-paper-inputs")
```

The launcher checks a local copy against `crb/paper_inputs.md5` as well.

With `CRB_OFFLINE=1` the launcher never contacts the Hub; the model, the inputs and the evaluation
datasets must then be cached. The evaluation datasets are pinned to fixed revisions, and scoring
always runs offline.

`crb/build_crb_inputs.sh` regenerates an input set on the CPU (see
[Building a new input set](crb/README.md#building-a-new-input-set)). Its operator files are
byte-identical to the paper's. The paper's var and literal files cannot be regenerated, because the
generator used for them depended on `PYTHONHASHSEED`. A rebuilt set is therefore a new sample of the
benchmark, and its numbers are not the paper's.

### Outputs

These go to `evaluation/crb/outputs/`, or to `--out_dir`:

- `results/<LABEL>_summary.json`, with these fields:
  - `headline_nr1_macro12`: `pass@1_T1` … `pass@1_T4`, `conf_gap`, `hit@1`, `hit@3`, `hit@5`,
    `conf_clean`, `conf_err`.
  - `macro`: `nr<n>_T<t>`, and `allnr_T<t>` for the mean over all n_replace.
  - `per_dataset_macro`: `<dataset>_T<t>`, the mean over error types and n_replace.
  - Every cell, `n_timeout_total`, and the run metadata.
- `results/<LABEL>_per_cell.csv` and `results/<LABEL>_per_sample.csv`.
- `runs/` (refined programs and per-step histories) and `logs/`.

Two scripts compare runs: `crb/crb_table.py` gives a table plus paired McNemar tests for any labels,
and `crb/crb_compare.py` gives a per-sample diff of two run trees. After `--purge` a run has no
refined files and no histories. `crb_compare.py` then compares the evaluated files (task id, prompt,
completion, step count, `test_passed`) and not the step-0 confidences or histories; it reports this
as `cells_from_evaluated_only`. It exits with status 1 if it compared no cell.

### Paper numbers

| Paper | Summary field | Printed | Rerun |
| --- | --- | --- | --- |
| Table 4, CDLM, T = 1 / 2 / 3 / 4 | `headline_nr1_macro12`, `pass@1_T1..T4` | 0.188 / 0.279 / 0.281 / 0.286 | 0.1883 / 0.2789 / 0.2814 / 0.2853 |
| Table 3, CDLM, τ = 0.9 | `headline_nr1_macro12`, `pass@1_T1` | 0.188 | 0.1883 |
| Table 7, CDLM: HumanEval / HumanEval+ / MBPP / MBPP+ / average | `per_dataset_macro` `<dataset>_T4`, `macro` `allnr_T4` (full grid) | 0.257 / 0.242 / 0.175 / 0.226 / 0.225 | 0.2569 / 0.2421 / 0.1749 / 0.2257 / 0.2249 |

The same run gives confidence gap 0.1636, Top-1 0.2277, Top-3 0.4499 and Top-5 0.6058 at
n_replace = 1.

**Table 4, T = 4 (0.2853 against the printed 0.286).** HumanEval/139, in the HumanEval+ literal
n_replace = 1 cell, runs for about 9 s against the 8 s execution timeout. It passed in the paper's
run and timed out in the reruns, with the same completion. Counted as passing, it gives 0.2858.
`n_timeout_total` in the summary counts timeouts, and scoring on a busy machine can produce more.

**Base and MDLM.** The launcher scores every model on the same pinned input set.
[`crb/README.md`](crb/README.md) lists the values it gives for the base model and MDLM-0.5B. These
are not the base and MDLM entries of Tables 1, 3 and 7. For those entries, the paper refined each
model's own input files, generated under that model's tag. Those files have the same operator
instances as the pinned set but different var and literal instances. Table 3's other thresholds are
not run either, because τ is fixed at 0.9.

### What was verified

All runs used one A100-PCIE-40GB with torch 2.5.0+cu121 and the CDLM-0.5B weights (sha256
`e43f9fa6…`).

- **Repository code, n_replace = 1** (48 cells, 12,852 refined programs):
  - Every refined program, step count, step-0 confidence and per-step history is bitwise identical
    to the paper's run.
  - `test_passed` differs for one program, HumanEval/139 (see above).
  - The per-cell CSV and the headline equal the reference rerun exactly.
- **Full grid** (240 cells, 37,028 refined programs):
  - This ran with a frozen copy of the pipeline. Its extra code, Fast-dLLM loading and padding for
    batch size > 1, is not used at batch size 1.
  - All completions and step-0 confidences are identical to the paper's run, and `test_passed`
    differs for the same single program.
  - n_replace 2..5 was not rerun with the repository files.
  - The base and MDLM values in `crb/README.md` come from the same frozen copy.
- **`build_crb_inputs.sh`** regenerates all 20 operator input files, raw and evaluated, byte for byte.

## Code generation

### Running it

```bash
# a Hub id at its pinned revision: 4 benchmarks x 2 decoders, workers spread over GPUs 0-3
bash evaluation/codegen/run_codegen_eval.sh \
    Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000@5a7170e7c2333e41d5312c690ac4818722a76c3a \
    outputs/codegen/cdlm 0,1,2,3

# a local HF checkpoint: HumanEval, vanilla decoder, one GPU with 4 workers
bash evaluation/codegen/run_codegen_eval.sh /path/to/checkpoint outputs/codegen/mine 0 4 vanilla humaneval

# spot check: the first 3 problems of each of the 4 virtual ranks
MAX_DOCS=3 bash evaluation/codegen/run_codegen_eval.sh <MODEL> outputs/codegen/spot 0 4 vanilla humaneval
```

The arguments are `MODEL OUT_DIR [GPUS=0] [PROCS_PER_GPU=4] [ALGS=vanilla,remdm] [TASKS=humaneval,humaneval_plus,mbpp,mbpp_plus]`.
`MODEL` is a local directory, a Hub id, or `hub_id@revision`. A Hub id without a revision is
resolved to its current commit, which is recorded with a warning. `PREFLIGHT_ONLY=1` runs only the
checks (see [Checks only](#checks-only-cpu)), and `--help` prints the usage.

The protocol:

- 10 samples per problem: 128 tokens in 128 steps, temperature 0.8, top-k 200.
- vanilla = `alg=p2_upgraded` and ReMDM = `alg=p2_upgraded_ReMDM`.
- The paper's 4-process lm-eval run is replayed as 4 virtual ranks, so the result does not depend on
  the number of GPUs.
- Before any GPU work, the launcher checks:
  - the 188 code files it runs, against [`codegen/MANIFEST.md5`](codegen/MANIFEST.md5);
  - the package versions (liger-kernel must be importable);
  - the `code_eval` metric;
  - the benchmark documents and prompts, loaded at pinned dataset revisions;
  - the model.

### Outputs

`OUT_DIR/summary.json` has these fields:

- `results.<alg>.<task>`: pass@1 and pass@10 with standard errors. `<alg>` is `p2_upgraded` or
  `p2_upgraded_ReMDM`.
- `he_avg_*`, `mbpp_avg_*` and `macro4_*`: averages over the benchmarks.
- `reference_check`: filled in when the weights are one of the three paper models (matched by
  sha256). It then compares every problem with the paper's recorded run.
- Provenance and package versions. `env.repo_commit` is the output of `git rev-parse HEAD` and
  `env.repo_describe` that of `git describe --always --dirty`.

`OUT_DIR` also holds `samples_<task>__<alg>.jsonl`, `metrics_<task>__<alg>.json` and `shards/`.
`codegen_eval.py compare` compares samples with a recorded run, and `codegen_eval.py paired` gives a
paired comparison of two models.

### Paper numbers

- **Table 5 (top):** `humaneval` and `humaneval_plus` pass@1 and pass@10, rounded to two decimals,
  for the three published models and both decoders.
- **Table 9:** `mbpp` and `mbpp_plus`. Its Avg column is `mbpp_avg_*`.

All 24 recorded values are in [`codegen/README.md`](codegen/README.md#recorded-results). One printed
value differs from its recorded run: Table 5 gives 0.42 for the CDLM-0.5B vanilla HumanEval
Pass@10, and the recorded run (and its rerun) gives 0.4146.

### What was verified

All runs used A100-PCIE-40GB GPUs.

- **Repository code, one GPU, 2 workers:**
  - The full CDLM-0.5B HumanEval vanilla run reproduces the recorded run exactly. All 1,640
    completions and the pass@1 and pass@10 of all 164 problems are identical (pass@1 0.2159,
    pass@10 0.4146). It took 22 min.
  - Two spot checks were also identical: CDLM-0.5B HumanEval vanilla (the first 3 problems of each
    rank) and Open-dCoder-0.5B MBPP+ ReMDM (the first 2).
- **A frozen copy of the same files** (byte-identical to `MANIFEST.md5`):
  - It reproduced three full recorded runs: CDLM-0.5B HumanEval vanilla, Open-dCoder-0.5B HumanEval
    vanilla and Open-dCoder-0.5B MBPP+ ReMDM. Its spot checks cover all three models, both decoders
    and all four benchmarks.
  - Rescoring the recorded completions of all 24 runs gives the recorded per-problem pass@1.
  - The other 21 model × decoder × benchmark runs were not repeated in full.

Other GPUs or library versions can change the sampled tokens. `reference_check` then reports how
many problems differ.

## Runtime

These times are per model, on one A100-PCIE-40GB.

| Run | GPU | CPU (scoring) |
| --- | --- | --- |
| CRB, full grid (240 cells, 37,028 programs) | about 35 min with 3 refinement jobs (about 0.6 GPU-hours) | about 45 min to 1 h with 8 jobs |
| CRB, `--nr 1` (48 cells, 12,852 programs) | about 12 min (estimated: about a third of the programs, 12,852 of 37,028) | 35 min measured with 6 jobs on a busy machine |
| Code generation, HumanEval, one decoder | about 20 min with 2 or 4 workers on an otherwise idle GPU (22 min measured; 32 min on a shared GPU) | included |
| Code generation, MBPP+, ReMDM | about 80 min with 4 workers | included |
| Code generation, 4 benchmarks × 2 decoders | about 6 h (estimated from the timings above, not measured) | included |

Notes on the table:

- MBPP and MBPP+ run at batch size 1 and take most of the code-generation time. On them, ReMDM is
  about twice as slow as vanilla.
- A HumanEval worker uses about 6 GB of GPU memory.
- Scoring executes every program on the CPU, and it slows down on a busy machine. Programs close to
  the time limit can then time out.
