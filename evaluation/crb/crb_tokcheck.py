#!/usr/bin/env python3
"""Tokenizer guard for a CRB input set.

Loads the tokenizer exactly as utils.load_model_and_tokenizer does
(AutoTokenizer, trust_remote_code=True, use_fast=False) and tokenizes, for every
test-failing sample of every input file, the refine prompt (function_head + "\\n")
and the buggy body with add_special_tokens=False, the same calls
refine_code.collate_fn_refine makes.  Prints a sha256 over all id lists plus the
pad and mask ids.  If this hash differs from the one of the tokenizer that
generated the inputs, error_positions (token indices) are misaligned.

Usage: crb_tokcheck.py <model_name_or_dir> <root containing buggy_datasets/> [--nr 1 2 3 4 5]
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

DATASETS = ["human-eval", "human-eval+", "mbpp", "mbpp+"]
ERROR_TYPES = ["operator", "var", "literal"]
PAPER_TAG = "Open-Dcoder-0.5B-mixture-mdm-step2000"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("inputs_root")
ap.add_argument("--nr", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--datasets", nargs="+", default=DATASETS)
ap.add_argument("--error_types", nargs="+", default=ERROR_TYPES)
ap.add_argument("--tag", default=PAPER_TAG)
ap.add_argument("--data_num", type=int, default=2)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, use_fast=False)
h = hashlib.sha256()
n = 0
bad_pos = 0
for nr in a.nr:
    for ds in a.datasets:
        for et in a.error_types:
            f = (Path(a.inputs_root) / "buggy_datasets" / ds / "evaluated" /
                 f"{a.tag}_{et}_{a.data_num}_wrong_{nr}_evaluated.jsonl")
            for line in open(f):
                r = json.loads(line)
                if r["test_passed"]:
                    continue
                ctx = tok(r["function_head"] + "\n", add_special_tokens=False)["input_ids"]
                body = tok(r["buggy_body"], add_special_tokens=False)["input_ids"]
                h.update(json.dumps([ctx, body]).encode())
                n += 1
                if any(not (0 <= p < len(body)) for p in (r.get("error_positions") or [])):
                    bad_pos += 1
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
out = {"model": a.model, "n_samples": n, "token_sha256": h.hexdigest(), "pad_id": pad,
       "id151665": tok.convert_ids_to_tokens(151665), "n_error_pos_out_of_range": bad_pos,
       "nr": a.nr}
print(json.dumps(out))
