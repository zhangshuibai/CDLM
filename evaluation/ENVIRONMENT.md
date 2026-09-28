# Evaluation environment

One Python environment runs both evaluations of the Open-dCoder-0.5B models in the paper:

- **CRB** (error localisation and correction): `refine_code.py`, `llada_sample.py`, `utils.py`,
  `evaluate_code.py`, `sanitize.py` at the repository root, plus `codecorrection/generate.py` to
  rebuild the benchmark;
- **code generation** (HumanEval, HumanEval+, MBPP, MBPP+ from scratch):
  `Open-dLLM/eval/eval_completion/eval.py` with the lm-evaluation-harness 0.4.9.1 vendored in
  `Open-dLLM/lm-evaluation-harness`.

`requirements.txt` in this directory pins the environment in which both evaluations reproduced the
paper's numbers bitwise. Training uses a different environment (see
[below](#not-the-training-environment)).

## Install

Linux x86_64, an NVIDIA driver for CUDA 12.1 or newer, and conda. No CUDA toolkit is needed.

```bash
conda create -n cdlm-eval python=3.11.13 -y && conda activate cdlm-eval
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r evaluation/requirements.txt
pip check
```

- **torch.** Use the CUDA 12.1 build (`2.5.0+cu121`). The bitwise-paired results (the CRB cells
  and the code-generation samples of the published models, reproduced bit for bit from the
  recorded runs) were produced with this build on A100-PCIE-40GB GPUs (driver 560.35.03). The
  CUDA 12.4 build that training uses is a different binary and was not checked for bitwise
  equality. The CUDA libraries come with the wheel (cuBLAS 12.1.3.1, cuDNN 9.1.0.70, NCCL 2.21.5).
- **flash-attn.** The URL is the official prebuilt wheel for torch 2.5, CUDA 12, Python 3.11 and
  the pre-C++11 ABI of the torch 2.5.0 wheels. `pip install flash-attn==2.7.4.post1
  --no-build-isolation` fails on machines whose `nvcc` is older than CUDA 11.7, because its
  `setup.py` checks `nvcc` before it looks for a prebuilt wheel (see
  [training/README.md](../training/README.md#environment)). Install it after torch.
- **Python 3.11.13.** Generated programs are executed by this interpreter during scoring.
- Do not `pip install` `Open-dLLM/` or `Open-dLLM/lm-evaluation-harness/`. Both are used from
  `PYTHONPATH` (next section). `Open-dLLM`'s `setup.py` would pull unpinned packages. An editable
  install of either one makes `import veomni` or `import lm_eval` resolve to the directory it was
  installed from whenever `PYTHONPATH` does not point elsewhere.

## Using it

From the repository root:

```bash
# CRB (refine_code.py, evaluate_code.py, codecorrection/generate.py)
export PYTHONPATH="$PWD/Open-dLLM"
python -c "import veomni; print(veomni.__file__)"      # <repo>/Open-dLLM/veomni/__init__.py

# code generation (eval.py and the vendored lm-eval)
export PYTHONPATH="$PWD/Open-dLLM/eval/eval_completion:$PWD/Open-dLLM/lm-evaluation-harness:$PWD/Open-dLLM"
python -c "import veomni, lm_eval; print(veomni.__file__, lm_eval.__file__)"
```

Scoring executes generated code, so it also needs `HF_ALLOW_CODE_EVAL=1`; read
[Executing generated code](#executing-generated-code) first.

## What the pins are for

| Packages | Why |
| --- | --- |
| `torch 2.5.0`, `triton 3.1.0`, `nvidia-nvjitlink-cu12 12.9.86` | Model forward pass. torch fixes the triton version; nvJitLink is an unversioned torch dependency. |
| `setuptools 78.1.1` | triton 3.1.0 imports it at start-up without declaring it. |
| `liger-kernel 0.5.8` | Required for identical numerics. When it can be imported, `veomni`'s Qwen2 replaces RMSNorm, the SwiGLU MLP and the rotary embedding with liger's triton kernels (`Open-dLLM/veomni/models/transformers/qwen2/modeling_qwen2.py`); without it the model runs the plain PyTorch layers. The kernels are compiled at run time, which needs a C compiler and the Python headers (conda's Python has them). |
| `flash-attn 2.7.4.post1` | Present in every verified run. It is not the attention kernel of the evaluated models: `veomni`'s Qwen2 is loaded without `attn_implementation`, so transformers uses SDPA. `import veomni.models` imports flash-attn only on machines with a GPU, from its vision-language and DeepSeek-V3 modules. Runs without it were not tested. |
| `transformers 4.54.1`, `tokenizers 0.21.4`, `safetensors 0.6.2`, `huggingface-hub 0.34.4`, `accelerate 1.10.1` | Model and tokenizer loading; `accelerate` also runs `eval.py`'s model wrapper. CRB error positions are token indices, so the tokenizer must give the same ids. |
| `regex 2025.9.1` | The slow `Qwen2Tokenizer`, which CRB loads (`use_fast=False`). |
| `numpy 2.3.2` | pass@k estimates and tensor conversion. |
| `torchdata`, `einops`, `pillow`, `packaging`, `psutil`, `PyYAML` | Imported by every `import veomni.models`. |
| `datasets 3.6.0`, `pandas 2.3.2`, `pyarrow 21.0.0`, `fsspec 2024.6.1`, `dill`, `multiprocess`, `xxhash` | Loading the benchmark datasets. datasets 3.6.0 does not bound pandas; without the pin pip installs pandas 3. |
| `evaluate 0.4.5` | Loads the `code_eval` metric that executes and scores the programs. |
| `autopep8`, `pycodestyle` | `codecorrection/generate.py` (re-indents MBPP code). |
| `peft`, `more-itertools`, `sacrebleu`, `Jinja2`, `MarkupSafe`, `tqdm` | Imported by the vendored lm-eval on the code-generation path (`lm_eval/models/hf_steered.py`, `lm_eval/models/vllm_causallms.py`, `lm_eval/api/metrics.py`, prompt templates). `lm_eval.models` imports `more_itertools` unconditionally; without the pin it resolves only to the copy vendored inside setuptools, and only after something has imported setuptools. |
| `filelock 3.13.1` and the rest of the last block | The remaining packages that a fresh install pulls in, at the verified versions. The filelock pin is required: filelock 3.32.3 (the release pip chose on 2026-09-28) raises `RuntimeError: os.fork is unsafe while filelock is changing descriptor ownership` when `code_eval`'s worker threads fork at the same time, and lm-eval's pass@k fails. |

With these pins a fresh install contains 72 packages (as listed by `pip freeze`), and every one
has the version of the verified environment.

Not installed, because nothing on the two paths needs them: the rest of lm-eval's declared
dependencies (`sqlitedict`, `jsonlines`, `word2number`, `rouge_score`, `pytablewriter`,
`tqdm-multiprocess`, `pybind11`, `zstandard`, `numexpr`, `scikit-learn`; lm-eval imports
scikit-learn only inside its F1 and MCC metrics). The verified environment also had
`scikit-learn`, `scipy`, `numexpr`, `zstandard` and `hf_transfer`. transformers, pandas, fsspec,
urllib3 and lm-eval import these only when they are installed, and nothing on these paths uses
them.

The upstream multi-process runner `Open-dLLM/eval/eval_completion/run_eval.sh` runs `eval.py` as a
script. That imports `eval_utils.py`, which needs `wandb`, and lm-eval's command line prints its
result table with `pytablewriter`. For that runner, also install `wandb==0.21.3
pytablewriter==1.2.1` (commented out at the end of `requirements.txt`).

## Not the training environment

Training has its own environment ([training/requirements.txt](../training/requirements.txt):
torch 2.5.0 for CUDA 12.4, transformers 4.57.0, tokenizers 0.22.2, datasets 4.2.0,
accelerate 1.11.0). The LLaDA-8B experiment also has its own
([training/llada8b_lora/requirements.txt](../training/llada8b_lora/requirements.txt)). Keep them
in separate environments:

- the transformers, tokenizers, datasets and accelerate pins conflict;
- `training/veomni` and `Open-dLLM/veomni` are both importable as `veomni` and they differ.
  Whichever directory comes first on `sys.path` hides the other. Never put `training/` on
  `PYTHONPATH` when evaluating, and do not install either copy into the evaluation environment.

## lm-evaluation-harness

The copy vendored in `Open-dLLM/lm-evaluation-harness` (0.4.9.1) works from `PYTHONPATH`; no
`pip install` is needed. The only use of installed package metadata is the lm-eval version string
written into result files, and that lookup is guarded. On the code-generation path, the loaded
lm-eval modules import these third-party packages at module level: `accelerate`, `datasets`,
`dill`, `evaluate`, `filelock`, `huggingface_hub`, `jinja2`, `more_itertools`, `numpy`,
`packaging`, `peft`, `requests`, `sacrebleu`, `torch`, `tqdm`, `transformers`, `yaml`. All of them
are pinned in `requirements.txt`.

`TaskManager` indexes the YAML files of every task directory, but only the `humaneval` and `mbpp`
task code (`utils.py`, `sanitize.py`) is imported.

## Hugging Face downloads

The code downloads these at run time and passes no revision, so it gets the Hub's `main`. The
verified runs used the revisions below. On 2026-09-28 the Hub's `main` of every dataset, of the
metric and of `fredzzp/open-dcoder-0.5B` was still this revision. The two `Shuibai12138` model
repositories have one later commit, which only adds a model card; the files the code loads are
unchanged.

| What | Loaded by | Hub repository | Revision |
| --- | --- | --- | --- |
| HumanEval | `evaluate_code.py`, `generate.py` as `openai_humaneval`; lm-eval task `humaneval` as `openai/openai_humaneval` | `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| HumanEval+ | both paths | `evalplus/humanevalplus` | `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` |
| MBPP | both paths, config `full` (the CRB scripts use the default config, which is `full`) | `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| MBPP+ | both paths | `evalplus/mbppplus` | `b2d74c91837c3f2a20c1299ae98133cbe7cfa077` |
| `code_eval` metric | `evaluate.load("code_eval")` in `evaluate_code.py` and in lm-eval's `humaneval/utils.py`, `mbpp/utils.py` | Space `evaluate-metric/code_eval` | tag `v0.4.0` (commit `da8da20af42f1f30de48bdc7cf4e57d5976a1b8d`) |
| CDLM-0.5B | `--model_name` (CRB), `pretrained=` (code generation) | `Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000` | `4581e1d4215a4055ccce6eaaf898f27276a8759c` |
| MDLM-0.5B | same | `Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000` | `b78b055f3a1f0e683d3893783c10fbfd939b0cf7` |
| Open-dCoder-0.5B (base) | same | `fredzzp/open-dcoder-0.5B` | `d0d86d5b99960c05258bb1f8265dd91564dbac67` |

Notes:

- **code_eval.** The verified runs used `code_eval.py` (git blob `0885712e698a34067e8faabe6b029ea8d719e024`,
  md5 `7dfb30101f10f0d4dc5ae2e0c33558f6`) and `execute.py` (blob
  `53517a805cf84758b858612b62794b874590159b`, md5 `4c2fcb60b139ea0a9f34f033f7ff2ecb`). evaluate
  caches them under `modules/evaluate_modules/metrics/evaluate-metric--code_eval/78d307ea938083398db7d9815f03ed661e9c15f60d77880ce007a8a02648f176/`.
  Tag `v0.4.0` and `main` both have these files. evaluate 0.4.5 first asks for tag `v0.4.5`,
  which the Space does not have, and then falls back to `main`, which can change. Setting
  `HF_SCRIPTS_VERSION=v0.4.0` makes it fetch the tag instead.
- **Model weights** (sha256 of `model.safetensors`): CDLM-0.5B
  `e43f9fa6b4cccfc18a2bac8925d64f5a020fa6a6d34db2c801220e22fdf8a680`, MDLM-0.5B
  `4f1f412cdac13553b76fd8de569ae9ed5a95447dfabaaf16da508d09344ff36e`, base
  `59c1a005f4b672bdd3bbdab6258b283dff3b87bab5cc4bbc0d479e8388f77b6e`.
  `Shuibai12138/CDLM-0.5B` has the CDLM weights, but its name lacks `open-dcoder`, so `utils.py`
  would not load it as a diffusion model. Use the `Open-Dcoder` name, or a local directory whose
  path contains `open-dcoder` and not `llada`.
- The CRB input files are not downloaded by the code; `refine_code.py` reads them from disk.

### Download once, then run offline

The launchers in `evaluation/crb/` and `evaluation/codegen/` handle the dataset revisions
themselves (see their READMEs). To run the scripts directly, download everything once with the
pinned revisions:

```bash
export HF_SCRIPTS_VERSION=v0.4.0 HF_ALLOW_CODE_EVAL=1
python - <<'EOF'
import evaluate
from datasets import load_dataset
from huggingface_hub import snapshot_download

for path, name, rev in [
    ("openai_humaneval", None, "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),         # CRB scripts
    ("openai/openai_humaneval", None, "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),  # lm-eval
    ("evalplus/humanevalplus", None, "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"),
    ("google-research-datasets/mbpp", "full", "4bb6404fdc6cacfda99d4ac4205087b89d32030c"),
    ("evalplus/mbppplus", None, "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"),
]:
    load_dataset(path, name, revision=rev)
evaluate.load("code_eval")
for repo in ["Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000",
             "Shuibai12138/Open-Dcoder-0.5B-baseline-mdm-step2000",
             "fredzzp/open-dcoder-0.5B"]:
    print(snapshot_download(repo))   # main; check model.safetensors against the sha256 above
EOF
```

Then run with

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

- Offline, datasets and evaluate load the most recently modified cached copy ("Using the latest
  cached version ..."). Keep one cached revision per dataset and one `code_eval` module, or check
  the path printed in the log.
- Both names of HumanEval must be cached: the CRB scripts load `openai_humaneval` and lm-eval
  loads `openai/openai_humaneval`. They are cached separately.
- Models are resolved through the cached `main`, so download them without a revision.
- If you set `HF_HOME`, use the same value for the download and for the runs.

## Executing generated code

> [!WARNING]
> Scoring runs model-generated programs, and the benchmark tests, on your machine without a
> sandbox. `code_eval`'s `execute.py` runs each program in a child process with a time limit and
> disables some functions (`os.kill`, `shutil.rmtree`, `subprocess.Popen`, ...). That is not
> isolation: the code runs as your user, with your files and your network. Run evaluations in a
> container or VM with no credentials, no network access and nothing you need to keep.

`code_eval` refuses to run unless `HF_ALLOW_CODE_EVAL=1`:

- `evaluate_code.py` sets it itself (at the top of the file);
- lm-eval's `humaneval` and `mbpp` task modules run a `code_eval` self-test when they are imported,
  so export `HF_ALLOW_CODE_EVAL=1` before running `eval.py` (its command line also requires
  `--confirm_run_unsafe_code`).

Each program has a time limit: 8 s per program with 100 threads in `evaluate_code.py` (its
`--timeout` and `--n_workers` defaults), and 3 s with 4 threads in the lm-eval tasks (the
`code_eval` defaults). A correct but slow program can time out on a heavily loaded machine, so do
not oversubscribe the CPU while scoring.

## How this environment was checked

- The modules of both paths were imported in the verified environment (CRB: the root scripts,
  `codecorrection/generate.py`, model and tokenizer loading, all four datasets and `code_eval`;
  code generation: `eval.py`'s model wrapper, the four lm-eval tasks, request building, filters
  and scoring; also `eval.py` run as a script). Every loaded module was mapped to its
  distribution. The results give the lists above.
- `pip install --dry-run -r evaluation/requirements.txt` in the verified environment installs
  nothing.
- A fresh environment built with the commands above on 2026-09-28 passes `pip check`. All
  72 packages have the verified versions. `libtorch_cuda.so` and flash-attn's
  `flash_attn_2_cuda` extension are byte-identical to the verified environment's. Both paths load
  the same third-party packages at the same versions, except optional ones that the verified
  environment happened to have (scikit-learn and its dependencies, `numexpr`, `zstandard`,
  `hf_transfer`, `chardet`, `protobuf`).
- CPU-only checks in that fresh environment:
  - the tokenizer ids of all 9257 pinned CRB samples hash to the value pinned by the CRB protocol;
  - `evaluate_code.py` on four pinned CRB input files (921 programs) reproduces the recorded
    pass/fail of every program;
  - the documents and prompts of the four lm-eval tasks hash to the values of the recorded paper
    runs;
  - re-scoring recorded paper samples through lm-eval (CDLM-0.5B vanilla on all four tasks,
    base ReMDM on HumanEval+, MDLM-0.5B vanilla on MBPP; 1870 problems) reproduces the recorded
    pass@1 and pass@10 of every problem.
- GPU generation was not re-run in the fresh environment.
