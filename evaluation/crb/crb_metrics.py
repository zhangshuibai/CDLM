#!/usr/bin/env python3
"""CRB metrics for one model run by run_crb.sh.

  * confidence c_i = p(z_i | z): probability of the currently visible token at
    body position i, read from history[...]['steps'][0]['conf_variable'] (the
    first forward pass over the unmodified buggy body).
  * per sample: gap = mean_{i not in E} c_i - mean_{i in E} c_i;
    hit@K = 1[E intersect (K lowest-confidence positions, stable argsort)].
    Samples without an in-range error position are skipped (none in the paper's set).
  * per cell: conf_gap = nanmean over samples; hit@K = mean over samples.
  * Pass@1 per cell = passed / n_eval from *_results_refined_evaluated.jsonl
    (cross-checked against pass_at_1_summary.json).
  * macro = unweighted mean over the (dataset x error_type) cells; the headline is
    the macro over the 12 cells at n_replace=1.  micro (sample-weighted) is also
    reported.

Localisation metrics do not depend on T; they are computed from every available
depth and the script checks that the step-0 confidences are bitwise identical.

Writes <outdir>/<label>_summary.json, <label>_per_cell.csv and
<label>_per_sample.csv.  Rows of the per-sample file pair exactly across models
evaluated on the same input set on (dataset, error_type, n_replace, T, idx).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

ALG = "self_conf-remask_vanilla_ct090_t00"
SETTING = "remove_all"
PAPER_TAG = "Open-Dcoder-0.5B-mixture-mdm-step2000"
DATASETS = ["human-eval", "human-eval+", "mbpp", "mbpp+"]
ERROR_TYPES = ["operator", "var", "literal"]
MAX_K = 6
MACRO_METRICS = ["pass@1", "conf_gap", "hit@1", "hit@2", "hit@3", "hit@4", "hit@5", "hit@6",
                 "conf_clean", "conf_err"]


def stem(tag, data_num, et, nr):
    return f"{tag}_{et}_{data_num}_wrong_{nr}_evaluated"


def cell_dir(root, prefix, kind, steps, ds, st):
    return (Path(root) / f"{prefix}_{kind}" / f"refined_steps{steps}" / SETTING / ALG /
            "buggy_datasets" / ds / "evaluated" / st)


def analyse_history(pt_file):
    """Localisation metrics of one cell (+ the raw step-0 confidences for the
    cross-depth identity check)."""
    hists = torch.load(pt_file, map_location="cpu", weights_only=False)
    conf_err, conf_clean, gaps = [], [], []
    hits = {k: 0 for k in range(1, MAX_K + 1)}
    n = 0
    raw = []
    for h in hists:
        if not h["steps"]:
            raw.append(None)
            continue
        c0 = h["steps"][0]["conf_variable"]
        raw.append(c0)
        conf = c0.float().numpy()
        err = [p for p in (h.get("error_positions") or []) if 0 <= p < len(conf)]
        if not err:
            continue
        n += 1
        err_set = set(err)
        clean_idx = [i for i in range(len(conf)) if i not in err_set]
        ce = float(np.mean(conf[err]))
        cc = float(np.mean(conf[clean_idx])) if clean_idx else float("nan")
        conf_err.append(ce)
        conf_clean.append(cc)
        gaps.append(cc - ce)
        order = np.argsort(conf, kind="stable")
        for k in range(1, MAX_K + 1):
            if err_set & set(order[:k].tolist()):
                hits[k] += 1
    if n == 0:
        return None, raw
    out = {"n_samples": n,
           "conf_clean": float(np.nanmean(conf_clean)),
           "conf_err": float(np.mean(conf_err)),
           "conf_gap": float(np.nanmean(gaps))}
    for k in range(1, MAX_K + 1):
        out[f"hit@{k}"] = hits[k] / n
    return out, raw


def read_eval(rdir, st):
    ev = rdir / f"{st}_results_refined_evaluated.jsonl"
    if not ev.exists():
        return None
    rows = [json.loads(l) for l in open(ev)]
    if not rows:
        return None
    passed = [bool(r.get("test_passed")) for r in rows]
    summ = None
    sf = rdir / "pass_at_1_summary.json"
    if sf.exists():
        d = json.load(open(sf))
        if isinstance(d, list) and d:
            d = d[-1]
        summ = float(d.get("pass_at_1")) if "pass_at_1" in d else None
    n_to = sum(1 for r in rows if r.get("test_result") == "timed out")
    return {"rows": rows, "passed": passed, "pass@1": sum(passed) / len(passed),
            "n_eval": len(passed), "summary_pass@1": summ, "n_timeout": n_to}


def macro(cells, keys):
    out = {"n_cells": len(cells)}
    for m in keys:
        v = [c[m] for c in cells if c.get(m) is not None and not (isinstance(c[m], float) and math.isnan(c[m]))]
        if not v:
            continue
        out[m] = float(np.mean(v))
        out[m + "_ncells"] = len(v)
        wkey = "n_eval" if m == "pass@1" else "n_samples"
        w = [c[wkey] for c in cells if c.get(m) is not None]
        out[m + "_micro"] = float(np.average(v, weights=w)) if sum(w) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="working directory of run_crb.sh (contains <prefix>_results/)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--nr", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--error_types", nargs="+", default=ERROR_TYPES)
    ap.add_argument("--tag", default=PAPER_TAG, help="file-name tag of the input set")
    ap.add_argument("--data_num", type=int, default=2)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--meta", default=None, help="JSON file with run metadata to embed")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells, missing, problems, sample_rows = [], [], [], []
    for nr in a.nr:
        for ds in a.datasets:
            for et in a.error_types:
                st = stem(a.tag, a.data_num, et, nr)
                loc_ref, raw_ref = None, None
                for steps in a.steps:
                    hp = cell_dir(a.root, a.prefix, "history", steps, ds, st) / "refined_rank0.pt"
                    rd = cell_dir(a.root, a.prefix, "results", steps, ds, st)
                    tag = f"{ds}/{et}/nr{nr}/steps{steps}"
                    if not hp.exists():
                        missing.append(tag + " [history]")
                        continue
                    loc, raw = analyse_history(hp)
                    if loc is None:
                        problems.append(tag + ": no sample with in-range error position")
                        continue
                    if loc_ref is None:
                        loc_ref, raw_ref = loc, raw
                    else:
                        same = len(raw) == len(raw_ref) and all(
                            (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
                            for x, y in zip(raw, raw_ref))
                        if not same:
                            problems.append(tag + ": step-0 confidences differ from the other depths")
                    ev = read_eval(rd, st)
                    if ev is None:
                        missing.append(tag + " [eval]")
                        p1, ne, nto = None, 0, None
                    else:
                        p1, ne, nto = ev["pass@1"], ev["n_eval"], ev["n_timeout"]
                        if ev["summary_pass@1"] is not None and abs(ev["summary_pass@1"] - p1) > 1e-12:
                            problems.append(tag + f": summary pass@1 {ev['summary_pass@1']} != jsonl {p1}")
                        if ne != loc["n_samples"]:
                            problems.append(tag + f": n_eval {ne} != n_loc {loc['n_samples']}")
                        for i, (r, ok) in enumerate(zip(ev["rows"], ev["passed"])):
                            sample_rows.append({"dataset": ds, "error_type": et, "n_replace": nr,
                                                "T": steps - 1, "idx": i, "task_id": r["task_id"],
                                                "test_passed": int(ok)})
                    c = dict(loc)
                    c.update({"dataset": ds, "error_type": et, "n_replace": nr,
                              "refined_steps": steps, "T": steps - 1, "pass@1": p1, "n_eval": ne,
                              "n_timeout": nto})
                    cells.append(c)

    # aggregates
    Ts = sorted({s - 1 for s in a.steps})
    agg = {}
    for nr in a.nr:
        for T in Ts:
            sel = [c for c in cells if c["n_replace"] == nr and c["T"] == T]
            if sel:
                agg[f"nr{nr}_T{T}"] = macro(sel, MACRO_METRICS)
    for T in Ts:
        sel = [c for c in cells if c["T"] == T]
        if sel:
            agg[f"allnr_T{T}"] = macro(sel, MACRO_METRICS)
    # per-dataset table (paper appendix): macro over the (error_type x n_replace) cells
    per_ds = {}
    for T in Ts:
        for ds in a.datasets:
            sel = [c for c in cells if c["T"] == T and c["dataset"] == ds]
            if sel:
                per_ds[f"{ds}_T{T}"] = macro(sel, ["pass@1"])

    expected = len(a.nr) * len(a.datasets) * len(a.error_types) * len(a.steps)
    headline = {}
    for T in Ts:
        k = f"nr1_T{T}"
        if k in agg:
            headline[f"pass@1_T{T}"] = agg[k].get("pass@1")
    if "nr1_T1" in agg:
        for m in ["conf_gap", "hit@1", "hit@3", "hit@5", "conf_clean", "conf_err"]:
            headline[m] = agg["nr1_T1"].get(m)

    # evaluator-noise indicator: samples that failed only by hitting the 8 s
    # per-program timeout of evaluate_code.py (some are genuine infinite loops,
    # some are borderline programs whose outcome depends on machine load)
    n_timeout_total = sum(c["n_timeout"] or 0 for c in cells)
    n_eval_total = sum(c["n_eval"] or 0 for c in cells)
    summary = {
        "label": a.prefix,
        "n_timeout_total": n_timeout_total, "n_eval_total": n_eval_total,
        "complete": (len(cells) == expected and not [m for m in missing if "[eval]" in m] and not problems),
        "n_cells_found": len(cells), "n_cells_expected": expected,
        "missing": missing, "problems": problems,
        "headline_nr1_macro12": headline,
        "macro": agg,
        "per_dataset_macro": per_ds,
        "cells": cells,
    }
    if a.meta and Path(a.meta).exists():
        summary["run_meta"] = json.load(open(a.meta))
    (outdir / f"{a.prefix}_summary.json").write_text(json.dumps(summary, indent=2))

    fields = (["label", "dataset", "error_type", "n_replace", "refined_steps", "T", "n_samples", "n_eval", "n_timeout",
               "conf_clean", "conf_err", "conf_gap"] + [f"hit@{k}" for k in range(1, MAX_K + 1)] + ["pass@1"])
    with open(outdir / f"{a.prefix}_per_cell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({"label": a.prefix, **{k: c.get(k) for k in fields if k != "label"}})
    with open(outdir / f"{a.prefix}_per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "dataset", "error_type", "n_replace", "T", "idx",
                                          "task_id", "test_passed"])
        w.writeheader()
        for r in sample_rows:
            w.writerow({"label": a.prefix, **r})

    print(f"[crb_metrics] {a.prefix}: {len(cells)}/{expected} cells, "
          f"missing={len(missing)}, problems={len(problems)}")
    for T in Ts:
        k = f"nr1_T{T}"
        if k in agg:
            g = agg[k]
            print(f"  n_replace=1 T={T}  macro over {g['n_cells']} cells  Pass@1={g.get('pass@1', float('nan')):.4f}  "
                  f"gap={g.get('conf_gap', float('nan')):.4f}  Top-1={g.get('hit@1', float('nan')):.4f}  "
                  f"Top-3={g.get('hit@3', float('nan')):.4f}  Top-5={g.get('hit@5', float('nan')):.4f}")
    for p in problems:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
