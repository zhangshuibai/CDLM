#!/usr/bin/env python3
"""Per-sample comparison of two CRB runs (e.g. a rerun against a reference run).

Both runs use the build_output_paths() layout (<root>/<prefix>_{results,history}/...).
Per cell it counts samples whose input, refined completion, actual_steps, step-0
confidences (bitwise), full refinement history (every step's tensors) or
test_passed differ.  Optionally compares two <label>_per_sample.csv files.

  crb_compare.py --new ROOT PREFIX --ref ROOT PREFIX [--nr 1] [--steps 2 3 4 5]
                 [--datasets ...] [--error_types ...] [--tag TAG] [--data_num 2]
                 [--new_csv A_per_sample.csv --ref_csv B_per_sample.csv] [--out cmp.json]
"""
import argparse
import csv
import json
from pathlib import Path

import torch

ALG = "remove_all/self_conf-remask_vanilla_ct090_t00"
PAPER_TAG = "Open-Dcoder-0.5B-mixture-mdm-step2000"
DATASETS = ["human-eval", "human-eval+", "mbpp", "mbpp+"]
ERROR_TYPES = ["operator", "var", "literal"]
KEYS = ["diff_input", "diff_completion", "diff_actual_steps", "diff_conf0_bitwise", "diff_history",
        "diff_test_passed", "diff_passed_same_completion"]


def cell(root, prefix, kind, steps, ds, st):
    return Path(root) / f"{prefix}_{kind}" / f"refined_steps{steps}" / ALG / "buggy_datasets" / ds / "evaluated" / st


def jl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else None


def same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape \
            and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def conf0(h):
    return h["steps"][0]["conf_variable"] if h["steps"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs=2, required=True, metavar=("ROOT", "PREFIX"))
    ap.add_argument("--ref", nargs=2, required=True, metavar=("ROOT", "PREFIX"))
    ap.add_argument("--nr", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--error_types", nargs="+", default=ERROR_TYPES)
    ap.add_argument("--tag", default=PAPER_TAG)
    ap.add_argument("--data_num", type=int, default=2)
    ap.add_argument("--new_csv", default=None)
    ap.add_argument("--ref_csv", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows, missing = [], []
    for nr in a.nr:
        for ds in a.datasets:
            for et in a.error_types:
                st = f"{a.tag}_{et}_{a.data_num}_wrong_{nr}_evaluated"
                for s in a.steps:
                    nd, rd = cell(*a.new, "results", s, ds, st), cell(*a.ref, "results", s, ds, st)
                    A, B = jl(nd / f"{st}_results_refined.jsonl"), jl(rd / f"{st}_results_refined.jsonl")
                    EA = jl(nd / f"{st}_results_refined_evaluated.jsonl")
                    EB = jl(rd / f"{st}_results_refined_evaluated.jsonl")
                    hn = cell(*a.new, "history", s, ds, st) / "refined_rank0.pt"
                    hr = cell(*a.ref, "history", s, ds, st) / "refined_rank0.pt"
                    tag = f"{ds}/{et}/nr{nr}/steps{s}"
                    if A is None or B is None:
                        missing.append(tag)
                        continue
                    r = {"ds": ds, "et": et, "nr": nr, "steps": s, "n": len(A), "n_ref": len(B),
                         "diff_input": sum(x["original_buggy_body"] != y["original_buggy_body"] or
                                           x["task_id"] != y["task_id"] for x, y in zip(A, B)),
                         "diff_completion": sum(x["refined_completion"] != y["refined_completion"] or
                                                x["completion"] != y["completion"] for x, y in zip(A, B)),
                         "diff_actual_steps": sum(x["actual_steps"] != y["actual_steps"] for x, y in zip(A, B))}
                    if hn.exists() and hr.exists():
                        HA = torch.load(hn, map_location="cpu", weights_only=False)
                        HB = torch.load(hr, map_location="cpu", weights_only=False)
                        r["n_hist"] = [len(HA), len(HB)]
                        r["diff_conf0_bitwise"] = sum(not same(conf0(x), conf0(y)) for x, y in zip(HA, HB)) + \
                            abs(len(HA) - len(HB))
                        r["diff_history"] = sum(not same(x, y) for x, y in zip(HA, HB)) + abs(len(HA) - len(HB))
                        d = [float((conf0(x).float() - conf0(y).float()).abs().max()) for x, y in zip(HA, HB)
                             if conf0(x) is not None and conf0(y) is not None and conf0(x).shape == conf0(y).shape]
                        r["max_abs_conf0_diff"] = max(d, default=0.0)
                    else:
                        missing.append(tag + " [history]")
                    if EA is not None and EB is not None:
                        r["diff_test_passed"] = sum(bool(x["test_passed"]) != bool(y["test_passed"]) for x, y in zip(EA, EB))
                        r["diff_passed_same_completion"] = sum(
                            bool(x["test_passed"]) != bool(y["test_passed"]) and x["completion"] == y["completion"]
                            for x, y in zip(EA, EB))
                        r["pass_new"] = sum(bool(x["test_passed"]) for x in EA) / len(EA)
                        r["pass_ref"] = sum(bool(x["test_passed"]) for x in EB) / len(EB)
                        r["passed_diff_task_ids"] = [x["task_id"] for x, y in zip(EA, EB)
                                                     if bool(x["test_passed"]) != bool(y["test_passed"])]
                    else:
                        missing.append(tag + " [eval]")
                    rows.append(r)

    tot = {"cells_compared": len(rows), "n": sum(r["n"] for r in rows),
           "n_len_mismatch": sum(r["n"] != r["n_ref"] for r in rows)}
    for k in KEYS:
        tot[k] = sum(r.get(k, 0) for r in rows)
    ev = [r for r in rows if "pass_new" in r]
    tot["cells_pass_equal"] = sum(abs(r["pass_new"] - r["pass_ref"]) < 1e-12 for r in ev)
    tot["max_abs_cell_pass_diff"] = max((abs(r["pass_new"] - r["pass_ref"]) for r in ev), default=None)
    tot["max_abs_conf0_diff"] = max((r.get("max_abs_conf0_diff", 0.0) for r in rows), default=None)
    out = {"totals": tot, "missing": missing, "cells": rows}

    if a.new_csv and a.ref_csv:
        def load(p):
            with open(p) as fh:
                return {(r["dataset"], r["error_type"], int(r["n_replace"]), int(r["T"]), int(r["idx"])):
                        (r["task_id"], int(r["test_passed"])) for r in csv.DictReader(fh)
                        if int(r["n_replace"]) in a.nr and r["dataset"] in a.datasets
                        and r["error_type"] in a.error_types and int(r["T"]) + 1 in a.steps}
        pa, pb = load(a.new_csv), load(a.ref_csv)
        common = pa.keys() & pb.keys()
        out["per_sample_csv"] = {
            "n_new": len(pa), "n_ref": len(pb), "n_common": len(common),
            "task_id_mismatch": sum(pa[k][0] != pb[k][0] for k in common),
            "test_passed_identical": sum(pa[k][1] == pb[k][1] for k in common),
            "test_passed_different": sorted([list(k) + [pa[k][0], pa[k][1], pb[k][1]] for k in common
                                             if pa[k][1] != pb[k][1]])}

    print(json.dumps(tot))
    if "per_sample_csv" in out:
        p = out["per_sample_csv"]
        print(json.dumps({k: v for k, v in p.items() if k != "test_passed_different"}))
        for d in p["test_passed_different"]:
            print("  CSV DIFF", d)
    for r in rows:
        if any(r.get(k) for k in KEYS):
            print("  DIFF", json.dumps({k: v for k, v in r.items() if k not in ("pass_new", "pass_ref")}))
    for m in missing:
        print("  MISSING", m)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
