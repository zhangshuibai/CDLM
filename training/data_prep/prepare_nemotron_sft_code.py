"""Download the training corpus of the Open-dCoder-0.5B CDLM / MDLM runs.

The paper runs train on the Nemotron-SFT-Code subset of
nvidia/Nemotron-Pretraining-SFT-v1 (78 parquet shards, 76,447,340 rows,
59.3 GB), pinned to the Hub revision that was downloaded for the paper. The
shards are consumed as-is: tasks/train_torch.py streams the parquet files
(--data.datasets_type=iterable), reads the `text` column (data_type=plaintext,
text_keys=text), tokenizes on the fly and packs 4096-token sequences. There is
no offline preprocessing step.

The dataset is gated: accept its terms on the Hub and log in
(`huggingface-cli login` or HF_TOKEN) before running.

    python data_prep/prepare_nemotron_sft_code.py                 # download + size check
    python data_prep/prepare_nemotron_sft_code.py --sha256        # also hash every shard
    python data_prep/prepare_nemotron_sft_code.py --paper_order   # see below

--paper_order: build_iterative_dataset() lists the data directory with
os.listdir(), whose order depends on the filesystem, and the streaming shuffle
permutes shards starting from that order. To feed the shards in exactly the
order of the paper runs, this option creates one directory per shard (symlink)
and writes a comma-separated --data.train_path to <local_dir>/train_path_paper_order.txt.
"""

import argparse
import hashlib
import os
import sys

from huggingface_hub import snapshot_download


REPO_ID = "nvidia/Nemotron-Pretraining-SFT-v1"
REVISION = "af7991c59eeb5e53a98bb6b1ee7a96cc3754eb39"
SUBSET = "Nemotron-SFT-Code"
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron_sft_code_files.tsv")
# <repo>/data, so that the shards land in <repo>/data/Nemotron-SFT-Code (the DATA_DIR default of scripts/)
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def read_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("file\t"):
                continue
            name, size, num_rows, sha256 = line.rstrip("\n").split("\t")
            rows.append((name, int(size), int(num_rows), sha256))
    return rows


def sha256_of(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.environ.get("DATA_ROOT", DEFAULT_ROOT))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--sha256", action="store_true", help="hash every shard (slow, ~59 GB)")
    parser.add_argument("--paper_order", action="store_true")
    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    data_dir = os.path.join(local_dir, SUBSET)
    if not args.skip_download:
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=[f"{SUBSET}/*.parquet"],
            local_dir=local_dir,
            max_workers=args.max_workers,
        )

    manifest = read_manifest()
    expected = {name for name, *_ in manifest}
    found = set(os.listdir(data_dir))
    if found != expected:
        sys.exit(
            f"{data_dir} must contain exactly the {len(expected)} shards and nothing else "
            f"(missing: {sorted(expected - found)[:5]}, unexpected: {sorted(found - expected)[:5]})"
        )
    for name, size, _, sha256 in manifest:
        path = os.path.join(data_dir, name)
        if os.path.getsize(path) != size:
            sys.exit(f"size mismatch for {path}")
        if args.sha256 and sha256_of(path) != sha256:
            sys.exit(f"sha256 mismatch for {path}")
    print(f"OK: {len(manifest)} shards in {data_dir}")

    train_path = data_dir
    if args.paper_order:
        order_root = os.path.join(local_dir, f"{SUBSET}-paper-order")
        dirs = []
        for i, (name, *_) in enumerate(manifest):
            shard_dir = os.path.join(order_root, f"{i:03d}")
            os.makedirs(shard_dir, exist_ok=True)
            link = os.path.join(shard_dir, name)
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(os.path.join(data_dir, name), shard_dir), link)
            dirs.append(shard_dir)
        train_path = ",".join(dirs)
        out = os.path.join(local_dir, "train_path_paper_order.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(train_path + "\n")
        print(f"paper shard order written to {out}")
        print(f'use: --data.train_path="$(cat {out})"')
    else:
        print(f"use: --data.train_path={train_path}")


if __name__ == "__main__":
    main()
