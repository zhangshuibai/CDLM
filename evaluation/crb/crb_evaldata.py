#!/usr/bin/env python3
"""Evaluation data used by evaluate_code.py: prefetch at the pinned revisions and fingerprint.

evaluate_code.py calls load_dataset(<name>) without a revision and
evaluate.load("code_eval").  run_crb.sh first caches the pinned revisions below
(`prefetch`), then runs every evaluation offline, and checks with `fingerprint`
that load_dataset(<name>) resolves to data whose test fields are identical to the
ones the paper's numbers were computed with.  The code_eval metric is fetched at
HF_SCRIPTS_VERSION (default v0.4.0, the verified module) and `fingerprint` reports whether it
matches the verified files (run_crb.sh stops if it does not).

  crb_evaldata.py prefetch    [--datasets ...]
  crb_evaldata.py fingerprint [--datasets ...]   (prints JSON; exit 1 on mismatch)
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
# evaluate 0.4.5 would ask for tag v0.4.5 of the code_eval Space (absent) and fall back to its main
os.environ.setdefault("HF_SCRIPTS_VERSION", "v0.4.0")

# name as used by evaluate_code.py, Hub revision, sha256 of the fields evaluate_code.py reads
PINNED = {
    "human-eval": ("openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                   "d5817fb0ff65a707dcb878015fabaac0512824f7c61f9d09dadf155469b29b90"),
    "human-eval+": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                    "501296e748fefaf91530cf8957fe1a9541e6949d07c87cd2375ced5aa43de93f"),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             "3be3947282d34db75adb582ddd78ccacd1977657289083d110dd21659ffc2a24"),
    "mbpp+": ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
              "0238d3bfbecda05cd1b36d1c9d8fced4e1f70dd5c11f26ed65298ceb33bd86a4"),
}
CODE_EVAL_SHA = "4fba54deb9b44025ac0facd5546758401d6d98e82e331253ab147daf3ddfd2a7"


def processed_info(dataset, ds):
    """The per-task fields evaluate_code.evaluate_main builds from the dataset."""
    info = {}
    for sample in ds["test"]:
        tid = str(sample["task_id"])
        if dataset in ["human-eval", "human-eval+"]:
            info[tid] = {"test": sample["test"], "entry_point": sample["entry_point"],
                         "prompt": sample["prompt"]}
        elif dataset == "mbpp":
            info[tid] = {"test_list": sample["test_list"], "test_setup_code": sample["test_setup_code"],
                         "prompt": sample["text"]}
        else:
            info[tid] = {"test_list": sample["test_list"], "test_setup_code": sample["test_imports"],
                         "prompt": sample["prompt"]}
    return info


def code_eval_sha():
    import evaluate
    m = evaluate.load("code_eval")
    d = os.path.dirname(inspect.getfile(type(m)))
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.endswith(".py") and f != "__init__.py":
            h.update(f.encode())
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefetch", "fingerprint"])
    ap.add_argument("--datasets", nargs="+", default=list(PINNED))
    a = ap.parse_args()
    from datasets import load_dataset
    if a.cmd == "prefetch":
        for d in a.datasets:
            name, rev, _ = PINNED[d]
            load_dataset(name, revision=rev)
            print(f"[crb_evaldata] cached {name}@{rev[:10]}")
        code_eval_sha()
        print("[crb_evaldata] cached code_eval metric")
        return
    out = {"datasets": {}, "mismatch": []}
    for d in a.datasets:
        name, rev, want = PINNED[d]
        info = processed_info(d, load_dataset(name))
        got = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
        out["datasets"][d] = {"name": name, "pinned_revision": rev, "n_tasks": len(info), "sha256": got}
        if got != want:
            out["mismatch"].append(d)
    out["code_eval_sha256"] = code_eval_sha()
    out["code_eval_matches"] = out["code_eval_sha256"] == CODE_EVAL_SHA
    out["hf_scripts_version"] = os.environ["HF_SCRIPTS_VERSION"]
    print(json.dumps(out))
    sys.exit(1 if out["mismatch"] else 0)


if __name__ == "__main__":
    main()
