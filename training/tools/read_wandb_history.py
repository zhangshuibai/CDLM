"""Print the full-precision training history of a run from its offline W&B files.

train.log rounds the loss, its components and grad_norm to 2 decimals. The trainer also logs
every step to W&B (rank 0; offline by default, under <run>/wandb/), with the values at full
precision. This script reads those files directly, without a W&B account or network access.
With WANDB_MODE=disabled no such record exists.

    python training/tools/read_wandb_history.py training/outputs/<run>
    python training/tools/read_wandb_history.py <run> --keys training/loss,losses/mdm --steps 1-2
    python training/tools/read_wandb_history.py <run>/wandb/offline-run-<...>/run-<id>.wandb --all --jsonl

PATH may be a run directory, its wandb/ directory, one offline-run-* directory or a .wandb
file. When several .wandb files are found (a resumed run), they are read in name order, which
is start-time order, and each starts with a "# <file>" line (TSV output only).

Default columns: step, training/loss, training/grad_norm, training/lr and every losses/* key.
Values are printed exactly as W&B stored them (JSON text); a missing value is empty. Needs the
wandb package of the training environment (requirements.txt: wandb==0.23.0).
"""

import argparse
import glob
import json
import os
import sys

DEFAULT_KEYS = ("training/loss", "training/grad_norm", "training/lr")


def find_files(path):
    if os.path.isfile(path):
        return [os.path.realpath(path)]
    if not os.path.isdir(path):
        sys.exit(f"not a file or directory: {path}")
    patterns = ["*.wandb", "*/*.wandb", "wandb/*/*.wandb"]
    files = {os.path.realpath(f) for p in patterns for f in glob.glob(os.path.join(path, p))}
    if not files:
        sys.exit(f"no .wandb file under {path} (was the run made with WANDB_MODE=disabled?)")
    return sorted(files, key=lambda f: (os.path.basename(os.path.dirname(f)), f))


def read_history(path):
    """Yields (step, {key: value_json}) for every history record of one .wandb file."""
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore
    except ImportError as e:
        sys.exit(f"cannot import the wandb file reader ({e}); use the training environment")
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except Exception as e:  # a record cut off when the process was killed
            print(f"WARNING: {path}: stopped at an unreadable record ({type(e).__name__}: {e})", file=sys.stderr)
            return
        if data is None:
            return
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        items = {(item.key or "/".join(item.nested_key)): item.value_json for item in record.history.item}
        step = int(items["_step"]) if "_step" in items else record.history.step.num
        yield step, items


def parse_steps(spec):
    if spec is None:
        return None
    lo, _, hi = spec.partition("-")
    try:
        lo = int(lo)
        hi = int(hi) if hi else lo
    except ValueError:
        sys.exit(f"--steps must be N or N-M (got {spec!r})")
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="run directory, wandb/ directory, offline-run-* directory or .wandb file")
    parser.add_argument("--keys", help="comma-separated keys to print instead of the default columns")
    parser.add_argument("--all", action="store_true", help="print every logged key")
    parser.add_argument("--steps", help="only steps N or N-M (inclusive)")
    parser.add_argument("--jsonl", action="store_true", help="one JSON object per step instead of TSV")
    args = parser.parse_args()
    steps = parse_steps(args.steps)

    for path in find_files(args.path):
        rows = [
            (step, items)
            for step, items in read_history(path)
            if steps is None or steps[0] <= step <= steps[1]
        ]
        present = sorted({k for _, items in rows for k in items if not k.startswith("_")})
        if args.keys:
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        elif args.all:
            keys = present
        else:
            keys = [k for k in DEFAULT_KEYS if k in present] + [k for k in present if k.startswith("losses/")]
        if args.jsonl:
            for step, items in rows:
                out = {"step": step}
                out.update({k: json.loads(items[k]) for k in keys if k in items})
                print(json.dumps(out))
        else:
            print(f"# {path}")
            print("\t".join(["step"] + keys))
            for step, items in rows:
                print("\t".join([str(step)] + [items.get(k, "") for k in keys]))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # output piped into head, etc.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
