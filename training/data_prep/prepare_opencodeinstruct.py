"""Build an ungated substitute for the paper's training corpus from nvidia/OpenCodeInstruct.

The paper runs train on Nemotron-SFT-Code (nvidia/Nemotron-Pretraining-SFT-v1), which is gated
and licensed for internal training only; see data_prep/README.md. nvidia/OpenCodeInstruct
(CC BY 4.0, ungated) stores instruction/solution pairs as separate `input` and `output` columns.
This script renders every row as one `text` string in the layout of the instruction/solution
documents of Nemotron-SFT-Code:

    text = "input: " + input + " output: " + output

Nothing precedes "input: " and nothing follows the solution (EOS is appended at tokenization).
The fields are used verbatim: no row is filtered, modified or reordered.

    python training/data_prep/prepare_opencodeinstruct.py            # download, check, render
    python training/data_prep/prepare_opencodeinstruct.py --sha256   # also hash every downloaded shard
    python training/data_prep/prepare_opencodeinstruct.py --shards 1 --local_dir <dir>   # first shard only

--shards N restricts every step to the first N shards in filename order: only they are downloaded,
checked and rendered, and train_path.txt lists only their directories. Use it for smoke tests,
in a separate --local_dir so that the full train_path.txt is not replaced. Each rendered shard
is identical to the same shard of a full run.

Layout under --local_dir (default <repo>/data):

    OpenCodeInstruct/data/train-000NN-of-00050.parquet     download, as on the Hub
    OpenCodeInstruct-text/0NN/train-000NN-of-00050.parquet  rendered shard: id, text, metadata
    OpenCodeInstruct-text/train_path.txt                    value for --data.train_path / TRAIN_PATH

build_iterative_dataset() lists every train_path directory with os.listdir(), whose order depends
on the filesystem, and it does not accept a file as a train_path entry. Each rendered shard
therefore sits alone in its own directory, and train_path.txt lists these directories, comma-
separated, in filename order, so the shard order seen by the streaming shuffle is fixed.

Re-running is safe: the download resumes, and finished shards (same source sha256, row count and
template) are skipped.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

# A local_dir download does not need hf_xet's chunk cache, which would keep a second copy of the
# data under ~/.cache/huggingface/xet.
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


REPO_ID = "nvidia/OpenCodeInstruct"
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
PREFIX = "input: "
SEPARATOR = " output: "
TEMPLATE = PREFIX + "{input}" + SEPARATOR + "{output}"
RAW_SUBDIR = "OpenCodeInstruct"
TEXT_SUBDIR = "OpenCodeInstruct-text"
TRAIN_PATH_FILE = "train_path.txt"
META_KEY = b"cdlm_render"
ROW_GROUP_SIZE = 10_000
# 2000 optimizer steps x global batch 12 x 4096 tokens
RUN_TOKENS = 2000 * 12 * 4096
# open-dcoder-0.5B tokens per character of rendered text, measured on the full corpus; only used
# when the tokenizer cannot be loaded
FALLBACK_TOKENS_PER_CHAR = 0.2554
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opencodeinstruct_files.tsv")
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        (
            "metadata",
            pa.struct(
                [
                    ("source", pa.string()),
                    ("revision", pa.string()),
                    ("domain", pa.string()),
                    ("generation_algorithm", pa.string()),
                ]
            ),
        ),
    ]
)


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


def render_info(path):
    """Render metadata stored in a finished shard, or None if the file is missing or unreadable."""
    try:
        pf = pq.ParquetFile(path)
        info = json.loads(pf.schema_arrow.metadata[META_KEY])
        info["num_rows"] = pf.metadata.num_rows
        return info
    except Exception:
        return None


def is_done(path, num_rows, sha256):
    info = render_info(path)
    return (
        info is not None
        and info.get("num_rows") == num_rows
        and info.get("source_sha256") == sha256
        and info.get("template") == TEMPLATE
    )


def render_shard(job):
    src, dst, tmp, name, num_rows, sha256 = job
    tables = []
    for batch in pq.ParquetFile(src).iter_batches(
        batch_size=ROW_GROUP_SIZE, columns=["id", "input", "output", "domain", "generation_algorithm"]
    ):
        if batch.column("input").null_count or batch.column("output").null_count:
            raise ValueError(f"{src}: null input or output")
        texts = [
            PREFIX + i + SEPARATOR + o
            for i, o in zip(batch.column("input").to_pylist(), batch.column("output").to_pylist())
        ]
        n = len(texts)
        metadata = pa.StructArray.from_arrays(
            [
                pa.array([REPO_ID] * n, pa.string()),
                pa.array([REVISION] * n, pa.string()),
                batch.column("domain").cast(pa.string()),
                batch.column("generation_algorithm").cast(pa.string()),
            ],
            fields=list(SCHEMA.field("metadata").type),
        )
        tables.append(
            pa.Table.from_arrays(
                [batch.column("id").cast(pa.string()), pa.array(texts, pa.string()), metadata], schema=SCHEMA
            )
        )
    rows = sum(t.num_rows for t in tables)
    if rows != num_rows:
        raise ValueError(f"{src}: expected {num_rows} rows, read {rows}")
    meta = {
        "source": REPO_ID,
        "revision": REVISION,
        "source_file": f"data/{name}",
        "source_sha256": sha256,
        "template": TEMPLATE,
        "rows": rows,
        "text_chars": sum(pc.sum(pc.utf8_length(t.column("text"))).as_py() for t in tables),
        "text_bytes": sum(pc.sum(pc.binary_length(t.column("text"))).as_py() for t in tables),
    }
    schema = SCHEMA.with_metadata({META_KEY: json.dumps(meta, sort_keys=True).encode()})
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for t in tables:
            writer.write_table(t.replace_schema_metadata(schema.metadata), row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp, dst)
    return name, rows


def estimate_tokens(shard_paths, tokenizer_name, sample_rows):
    """Tokens per character of `text`, as counted by the training transform (without EOS)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:  # offline, not cached, or transformers missing
        print(f"tokenizer {tokenizer_name} not available ({type(e).__name__}); using the recorded ratio")
        return FALLBACK_TOKENS_PER_CHAR, "recorded ratio"
    chars = tokens = 0
    for path in shard_paths:
        pf = pq.ParquetFile(path)
        texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()[:sample_rows]
        chars += sum(len(t) for t in texts)
        tokens += sum(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    return tokens / chars, f"{tokenizer_name}, first {sample_rows} rows of every shard"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--local_dir", default=os.environ.get("DATA_ROOT", DEFAULT_ROOT))
    parser.add_argument("--max_workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--num_proc", type=int, default=min(8, os.cpu_count() or 1), help="parallel renders")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--sha256", action="store_true", help="hash every downloaded shard (6.9 GB)")
    parser.add_argument("--tokenizer", default="fredzzp/open-dcoder-0.5B", help="for the token-count estimate")
    parser.add_argument("--estimate_rows", type=int, default=500, help="rows per shard tokenized for the estimate")
    parser.add_argument(
        "--shards",
        type=int,
        metavar="N",
        help="only the first N shards in filename order (e.g. 1 for a smoke test): download, check and render "
        "those, and list only them in train_path.txt",
    )
    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    raw_root = os.path.join(local_dir, RAW_SUBDIR)
    raw_dir = os.path.join(raw_root, "data")
    text_dir = os.path.join(local_dir, TEXT_SUBDIR)
    manifest = read_manifest()
    known = {name for name, *_ in manifest}
    if [name for name, *_ in manifest] != sorted(known):
        sys.exit(f"{MANIFEST} is not in filename order")
    if args.shards is not None:
        if not 1 <= args.shards <= len(manifest):
            parser.error(f"--shards must be between 1 and {len(manifest)} (got {args.shards})")
        manifest = manifest[: args.shards]

    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=(
                ["data/*.parquet"] if args.shards is None else [f"data/{name}" for name, *_ in manifest]
            ),
            local_dir=raw_root,
            max_workers=args.max_workers,
        )

    expected = {name for name, *_ in manifest}
    # With --shards, the other shards of the manifest may be present (e.g. from a full download).
    allowed = expected if args.shards is None else known
    found = set(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else set()
    if expected - found or found - allowed:
        what = (
            f"exactly the {len(expected)} shards and nothing else"
            if args.shards is None
            else f"the first {len(expected)} shards and no file outside the manifest"
        )
        sys.exit(
            f"{raw_dir} must contain {what} "
            f"(missing: {sorted(expected - found)[:5]}, unexpected: {sorted(found - allowed)[:5]})"
        )
    for name, size, _, sha256 in manifest:
        path = os.path.join(raw_dir, name)
        if os.path.getsize(path) != size:
            sys.exit(f"size mismatch for {path}")
        if args.sha256 and sha256_of(path) != sha256:
            sys.exit(f"sha256 mismatch for {path}")
    print(f"OK: {len(manifest)} source shards in {raw_dir} ({REPO_ID}@{REVISION})")

    # Shards are written here first and moved into place when complete, so an interrupted run
    # never leaves a partial file where build_iterative_dataset() would list it.
    partial_dir = os.path.join(text_dir, ".partial")
    shutil.rmtree(partial_dir, ignore_errors=True)
    shard_dirs, shard_paths, jobs = [], [], []
    for i, (name, _, num_rows, sha256) in enumerate(manifest):
        shard_dir = os.path.join(text_dir, f"{i:03d}")
        os.makedirs(shard_dir, exist_ok=True)
        dst = os.path.join(shard_dir, name)
        other = sorted(set(os.listdir(shard_dir)) - {name})
        if other:
            sys.exit(f"{shard_dir} must hold only {name}; remove {other}")
        shard_dirs.append(shard_dir)
        shard_paths.append(dst)
        if not is_done(dst, num_rows, sha256):
            jobs.append((os.path.join(raw_dir, name), dst, os.path.join(partial_dir, name), name, num_rows, sha256))
    print(f"rendering {len(jobs)} shard(s), {len(manifest) - len(jobs)} already done")
    if jobs:
        os.makedirs(partial_dir)
        with ProcessPoolExecutor(max_workers=max(1, min(args.num_proc, len(jobs)))) as pool:
            for name, rows in pool.map(render_shard, jobs):
                print(f"  {name}: {rows} rows", flush=True)
        os.rmdir(partial_dir)

    infos = [render_info(p) for p in shard_paths]
    for (name, _, num_rows, sha256), path in zip(manifest, shard_paths):
        if not is_done(path, num_rows, sha256):
            sys.exit(f"{path} is incomplete; re-run this script")
    rows = sum(x["rows"] for x in infos)
    chars = sum(x["text_chars"] for x in infos)
    text_bytes = sum(x["text_bytes"] for x in infos)
    file_bytes = sum(os.path.getsize(p) for p in shard_paths)
    print(f"OK: {len(shard_paths)} rendered shards in {text_dir}")
    print(f"  rows {rows:,}, text {chars:,} characters / {text_bytes:,} bytes, parquet {file_bytes:,} bytes")

    ratio, how = estimate_tokens(shard_paths, args.tokenizer, args.estimate_rows)
    tokens = chars * ratio + rows  # + one EOS per document
    print(
        f"  estimated tokens: {tokens / 1e9:.2f}B ({ratio:.4f} tokens/char, {how}); "
        f"a {RUN_TOKENS / 1e6:.1f}M-token 2000-step run uses {RUN_TOKENS / tokens:.2%} of one pass"
    )

    out = os.path.join(text_dir, TRAIN_PATH_FILE)
    try:
        with open(out, encoding="utf-8") as f:
            previous = f.read().strip().split(",")
    except OSError:
        previous = None
    if previous is not None and previous != shard_dirs:
        print(f"WARNING: replacing {out}, which listed {len(previous)} shard(s), with {len(shard_dirs)}")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(",".join(shard_dirs) + "\n")
    os.replace(tmp, out)
    subset = "" if args.shards is None else f", first {len(shard_dirs)} of {len(known)} shards only (--shards)"
    print(f"shard order (filename order{subset}) written to {out}")
    print(f"use: OCI_TRAIN_PATH_FILE={out} bash training/scripts/train_0.5b_opencodeinstruct.sh")


if __name__ == "__main__":
    main()
