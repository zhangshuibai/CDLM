# Training data

The Open-dCoder-0.5B CDLM and MDLM runs continue training `fredzzp/open-dcoder-0.5B`
on the **Nemotron-SFT-Code** subset of
[`nvidia/Nemotron-Pretraining-SFT-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1).

| | |
|---|---|
| Hub repo | `nvidia/Nemotron-Pretraining-SFT-v1` (dataset, gated) |
| Config / split | `Nemotron-SFT-Code` / `train` (files `Nemotron-SFT-Code/*.parquet`) |
| Revision | `af7991c59eeb5e53a98bb6b1ee7a96cc3754eb39` (the Hub `main` branch has moved since) |
| Shards | 78 parquet files, 76,447,340 rows, 59,260,834,163 bytes |
| Columns | `id`, `text`, `metadata {category, models_used}`; only `text` is used |
| License | NVIDIA Data Agreement for Model Training, linked on the Hub page as "NVIDIA Open Data License Agreement" |
| Access | Gated with manual approval: request access on the Hub page (accepting the agreement), and NVIDIA reviews each request manually. Needed only to retrain the paper's own checkpoints. |

No preprocessing is applied. The parquet shards are passed to `tasks/train_torch.py` as they
are, with the settings from `configs/pretrain/qwen2_5_coder_500M.yaml` plus
`--data.datasets_type=iterable`:

- `datasets.load_dataset("parquet", ..., streaming=True)`, then `.shuffle(seed, buffer_size=10_000)`
  and `split_dataset_by_node` across data-parallel ranks;
- each record's `text` is tokenized without special tokens, followed by EOS, and cut into chunks
  of at most `max_seq_len=4096` tokens (`data_type: plaintext`, `text_keys: text`);
- chunks are packed into micro-batches of `micro_batch_size * 4096` tokens
  (`rmpad_with_pos_ids: true`).

`--data.train_path` must point to a directory that holds only the shards (no README or cache
files), which is what `Nemotron-SFT-Code/` looks like after the download below.

## Download

Commands in this file are run from the repository root, as in [`../README.md`](../README.md). The
scripts find their own paths, so they work from any directory. Only relative paths you pass are
resolved against the current directory.

The download needs a Hub token of an account that has been granted access (the Access row above).

```bash
huggingface-cli login            # or export HF_TOKEN=...
python training/data_prep/prepare_nemotron_sft_code.py [--local_dir $DATA_ROOT] [--sha256]
# default local_dir: <repo>/data, i.e. shards in <repo>/data/Nemotron-SFT-Code
# then: DATA_DIR=$DATA_ROOT/Nemotron-SFT-Code, i.e. --data.train_path=$DATA_ROOT/Nemotron-SFT-Code
```

The script downloads the pinned revision and checks file names and sizes (and sha256 with
`--sha256`) against `nemotron_sft_code_files.tsv`.

## Shard order

`build_iterative_dataset` (in `veomni/data/dataset.py`) lists the data directory with
`os.listdir()`, which is not sorted and depends on the filesystem, and the streaming shuffle
permutes shards starting from that list. The same seed on a directory listed in a different
order therefore yields a different data stream. `nemotron_sft_code_files.tsv` records the order
seen on the machine that ran the paper's runs. To reproduce it elsewhere without touching the
training code:

```bash
python training/data_prep/prepare_nemotron_sft_code.py --local_dir $DATA_ROOT --skip_download --paper_order
# --data.train_path="$(cat $DATA_ROOT/train_path_paper_order.txt)"
```

This creates one directory per shard (a symlink) and a comma-separated `train_path` in the
recorded order. It is a list of directories, not a single directory, so it has to reach
`--data.train_path` unchanged. On CPU, feeding this path through the real data pipeline of this snapshot
reproduces the first packed micro-batch logged by the paper's multi-seed runs (2 data-parallel
ranks, `micro_batch_size=3`, `global_batch_size=12`): seed 1234 on ranks 0 and 1 and seed 42 on
rank 0 match in length and in their first and last tokens. A filename-sorted order does not.
The data stream also depends on the number of data-parallel ranks and on the pinned `datasets`
version.

## Ungated substitute: OpenCodeInstruct

Nemotron-SFT-Code cannot be used where the training data itself has to be public. The Hub repo
is gated with manual approval. Its gate requires accepting the NVIDIA Data Agreement for Model
Training, which makes the data available "solely for the purpose of internal training of
Company AI Solutions" (clause 2.1) and forbids to "sell, rent, sublicense, transfer, distribute,
sublicence, publicly display, publicly perform or otherwise make available to others the
Datasets" (clause 2.2.2).
For such uses the release supports
[`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) instead.

**Models trained on OpenCodeInstruct are not the paper's models, and their results are not the
paper's results.** No number in the paper comes from this data. Two reference checkpoints trained
on it are released:
- `Shuibai12138/Open-Dcoder-0.5B-CDLM-OpenCodeInstruct` at revision
  `8eb87fe2ab6850ced7606c8a678c84f1b28fd170`;
- `Shuibai12138/Open-Dcoder-0.5B-MDLM-OpenCodeInstruct` at revision
  `535b36930a32109c87986c341bf554cc42e76b2e`.

They were trained with `ARM=cdlm` and `ARM=mdlm bash training/scripts/train_0.5b_opencodeinstruct.sh`
at commit 5e52812. See
[`../README.md`](../README.md#opencodeinstruct-reference-checkpoints) for their sha256 and
reference results.

| | |
|---|---|
| Hub repo | `nvidia/OpenCodeInstruct` (dataset, not gated) |
| Revision | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` |
| Shards | 50 parquet files `data/train-000NN-of-00050.parquet`, 5,000,000 rows, 6,861,113,102 bytes |
| Columns | `id`, `input`, `output`, `domain`, `generation_algorithm`, `llm_judgement`, `unit_tests`, `tests_execution_status`, `average_test_score` |
| License | CC BY 4.0 |

OpenCodeInstruct ([Ahmad et al., 2025](https://arxiv.org/abs/2504.04030)) holds Python
programming questions with solutions. According to its technical report, the seed questions
come from Python functions in The Stack v2 (turned into tasks by Qwen2.5-32B-Instruct with
OSS-Instruct) and from TACO; the questions were expanded with self-instruct (2,653,523 rows) and
evol-instruct (2,346,477 rows); the solutions were generated by Qwen2.5-Coder-32B-Instruct, the
model behind 34.8% of the Nemotron-SFT-Code documents; and the questions were n-gram
decontaminated against evaluation benchmarks. Nothing is filtered, deduplicated or
decontaminated on top of that: all 5,000,000 rows are used, in source order. That includes the
3,358,695 rows (67%) whose solution fails at least one of the dataset's own LLM-generated unit tests
(`average_test_score < 1`) and 23,451 rows whose rendered text repeats an earlier row.

### Rendering

The trainer reads one `text` column (`data_type: plaintext`), while OpenCodeInstruct stores
`input` and `output` separately. `prepare_opencodeinstruct.py` joins them in the layout of the
paper corpus:

```
text = "input: " + input + " output: " + output
```

This is how Nemotron-SFT-Code renders an instruction/solution pair, as measured on the pinned
revision:
- 63,227,734 of its 76,447,340 documents (82.7%) are such pairs: every document from
  Qwen2.5-Coder-32B-Instruct (26,587,723) and from Mixtral-8x22B-v0.1 (36,640,011). The rest,
  from DeepSeek-R1 and DeepSeek-R1-0528, use a different layout, often with `<think>` traces,
  and have no counterpart in OpenCodeInstruct.
- A pair document starts with `input: `; nothing precedes it.
- Instruction and solution are separated by ` output: `: one space on each side, no newline.
  60,838,722 pair documents (96.2%) contain it. The other 2,389,012 lack it, mostly because
  there is no space before `output:`; in a 6-shard sample, all of them are Mixtral documents.
  This irregularity is not reproduced.
- The solution ends the document. No closing template or suffix follows it, and the EOS token
  is appended at tokenization, in both corpora.

In a sample of 149,622 Qwen2.5-Coder documents from 6 shards:
- every document has the ` output: ` separator;
- no instruction starts or ends with whitespace;
- one solution starts with whitespace;
- 89% of the solutions start with a code fence and 95% end with one;
- 9 documents end with a newline.

On the OpenCodeInstruct side, no `input` or `output` starts or ends with whitespace, and every
`output` is exactly one ```` ```python ```` block. The fields are therefore used verbatim, and
every rendered document has the boundary structure of a typical Qwen2.5-Coder document of the
paper corpus: `input: ` at offset 0, ` output: ```python` at the boundary, and ```` ``` ```` as
the last characters. The content still differs:
- OpenCodeInstruct solutions indent with 4 spaces, while about 39% of the sampled
  Qwen2.5-Coder documents of the paper corpus indent with tabs.
- Its questions are often structured Markdown (`**Input:**`, `**Sample Input:**` with fenced
  examples).
- It is Python only.

The two corpora share no document. Over both full pinned revisions (64-bit hashes), no
Nemotron-SFT-Code document equals a rendered OpenCodeInstruct text, and no Nemotron-SFT-Code
instruction equals an OpenCodeInstruct `input`: 0 exact matches. The only overlap is in
solutions: 239 Nemotron-SFT-Code solutions equal an OpenCodeInstruct `output`. These are generic
short functions (for example, the area of a rectangle) paired with different instructions.

### Size

The 5,000,000 rendered documents hold 9,384,931,631 characters. With the `open-dcoder-0.5B`
tokenizer they come to 2,402,350,725 tokens, counting one EOS per document. No document is
longer than 4096 tokens, so none is split. A 2000-step run reads at most
2000 × 12 × 4096 = 98.3M tokens, about 4% of one pass, so the data is never repeated within a
run. The download takes 6.9 GB, and the rendered shards take 2.7 GB.

### Preparation and training

```bash
python training/data_prep/prepare_opencodeinstruct.py [--local_dir $DATA_ROOT] [--sha256] [--shards N]
# default local_dir: <repo>/data; no Hub login needed
bash training/scripts/train_0.5b_opencodeinstruct.sh
# same settings and variables as train_0.5b.sh (ARM, SEED, STOP_STEP, NPROC, ...),
# run name open-dcoder-0.5B-oci-<ARM>-seed<SEED>-step<STOP_STEP>
```

The script downloads the pinned revision into `$DATA_ROOT/OpenCodeInstruct/data/`. It checks
file names and sizes against `opencodeinstruct_files.tsv`, and also sha256 with `--sha256`. It
turns off `hf_xet`'s chunk cache unless `HF_XET_CHUNK_CACHE_SIZE_BYTES` is set; otherwise the
cache would hold a second copy of the data under `~/.cache/huggingface/xet`.

It then writes one rendered shard per input shard, each in its own directory:
`$DATA_ROOT/OpenCodeInstruct-text/0NN/train-000NN-of-00050.parquet`. Each shard has the columns
`id`, `text` and `metadata {source, revision, domain, generation_algorithm}`, and only `text` is
used for training. Finally it writes `$DATA_ROOT/OpenCodeInstruct-text/train_path.txt`, which
lists these directories, comma-separated, in filename order. This is the mechanism of
`--paper_order` above, with the files in place of symlinks: each directory holds one file, so the
shard order does not depend on the filesystem.

Pass `train_path.txt` as `TRAIN_PATH` (the preset does this; `OCI_TRAIN_PATH_FILE` overrides
its location). `OpenCodeInstruct-text/` itself is not a valid `DATA_DIR`, because it contains
directories. The data stream still depends on the seed, the number of data-parallel ranks and
the `datasets` version. Re-running the script skips finished shards; delete a shard's file to
render it again.

`--shards N` restricts every step to the first N shards in filename order. The script downloads,
checks and renders only those shards, and writes a `train_path.txt` that lists only them. Use it
for smoke tests, with a separate `--local_dir` so that a full `train_path.txt` is not replaced (the
script warns when it replaces a `train_path.txt` that listed other shards). Then point the launcher
at that file:

```bash
python training/data_prep/prepare_opencodeinstruct.py --shards 1 --local_dir data/oci-smoke
OCI_TRAIN_PATH_FILE=data/oci-smoke/OpenCodeInstruct-text/train_path.txt STOP_STEP=2 \
    bash training/scripts/train_0.5b_opencodeinstruct.sh
```

With `--shards 1` the download is 133 MB and the rendering takes a few seconds. Without a Hub token
and with a fresh cache, the rendered shard was identical (`pyarrow.Table.equals`, and byte-identical
as a file) to shard 000 of a full run. The launcher prints `Train path: TRAIN_PATH (1 entries, ...)`.

On CPU, the rendered shards were fed through the real data pipeline of this snapshot
(`build_iterative_dataset` and `build_dataloader` with the CDLM collator, `micro_batch_size=3`)
via `train_path.txt`. It yields packed micro-batches of up to 3 × 4096 tokens, for example
12,266 tokens holding 25 documents, each ending in EOS. Their first tokens are `input` and `:`,
as in the first micro-batch logged by the paper's runs.

### License and attribution

OpenCodeInstruct is made available by NVIDIA Corporation under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
This repository does not redistribute it; the script downloads it from the Hub. The rendered
shards are an adapted version of it. If you share them, keep a notice such as:

> Contains data from OpenCodeInstruct by NVIDIA Corporation
> (https://huggingface.co/datasets/nvidia/OpenCodeInstruct, revision
> 8f3ba5bafe4d6e8db46082cf7ae6741bc370604d), licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: the `input` and `output` fields of
> each row were joined into one `text` field as "input: {input} output: {output}", and all
> columns other than `id`, `domain` and `generation_algorithm` were dropped.

When you report results from models trained on it, cite the dataset:

```bibtex
@article{ahmad2025opencodeinstruct,
  title={OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs},
  author={Wasi Uddin Ahmad and Aleksander Ficek and Mehrzad Samadi and Jocelyn Huang and Vahid Noroozi and Somshubra Majumdar and Boris Ginsburg},
  year={2025},
  eprint={2504.04030},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.04030},
}
```
