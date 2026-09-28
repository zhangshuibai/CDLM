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
| License | NVIDIA Open Data License Agreement (accept it on the Hub first) |

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

```bash
cd training                      # run from the training/ directory
huggingface-cli login            # or export HF_TOKEN=...
python data_prep/prepare_nemotron_sft_code.py [--local_dir $DATA_ROOT] [--sha256]
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
python data_prep/prepare_nemotron_sft_code.py --local_dir $DATA_ROOT --skip_download --paper_order
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
