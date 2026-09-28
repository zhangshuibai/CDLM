#!/usr/bin/env python3
"""Cross-model CRB table + exact-paired tests for models evaluated by run_crb.sh.

Every model is run on the same input set, so rows of <label>_per_sample.csv pair
exactly on (dataset, error_type, n_replace, T, idx): same buggy program, same task.

Usage: crb_table.py --results_dir DIR --labels base mdlm cdlm [--pairs cdlm:mdlm ...]
                    [--nr 1] [--out table.json]
Prints the macro over the (dataset x error_type) cells at the chosen n_replace
levels: Pass@1 at T=1..4, confidence gap, Top-1/3/5; then McNemar exact
(two-sided binomial on discordant pairs) pooled over the chosen cells, per T.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    try:
        from scipy.stats import binom
        return float(min(1.0, 2 * binom.cdf(k, n, 0.5)))
    except Exception:
        return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--pairs", nargs="*", default=[])
ap.add_argument("--nr", type=int, nargs="+", default=[1])
ap.add_argument("--out", default=None)
ap.add_argument("--exclude", nargs="*", default=[],
                help="dataset:task_id items to drop from the paired tests, e.g. "
                     "human-eval+:HumanEval/139 (runs ~9 s against the 8 s timeout, so its "
                     "outcome can flip between identical completions)")
a = ap.parse_args()
EXCL = {tuple(x.split(":", 1)) for x in a.exclude}
R = Path(a.results_dir)
out = {"nr": a.nr, "models": {}, "paired": {}}

print(f"n_replace in {a.nr}; macro over (dataset x error_type x n_replace) cells")
print(f"{'model':14s} " + " ".join(f"P@1 T={t}" for t in range(1, 5)) + "   gap     Top-1   Top-3   Top-5")
for lab in a.labels:
    s = json.load(open(R / f"{lab}_summary.json"))
    cells = [c for c in s["cells"] if c["n_replace"] in a.nr]
    row = {}
    for T in range(1, 5):
        v = [c["pass@1"] for c in cells if c["T"] == T and c["pass@1"] is not None]
        row[f"pass@1_T{T}"] = float(np.mean(v)) if v else None
        row[f"n_cells_T{T}"] = len(v)
    loc = [c for c in cells if c["T"] == min(c2["T"] for c2 in cells)]
    for m in ["conf_gap", "hit@1", "hit@3", "hit@5", "conf_clean", "conf_err"]:
        row[m] = float(np.mean([c[m] for c in loc]))
    row["complete"] = s["complete"]
    out["models"][lab] = row
    f = lambda x: "  --  " if x is None else f"{x:.4f}"
    print(f"{lab:14s} " + " ".join(f"{f(row[f'pass@1_T{T}']):>8s}" for T in range(1, 5)) +
          f"  {row['conf_gap']:.4f}  {row['hit@1']:.4f}  {row['hit@3']:.4f}  {row['hit@5']:.4f}"
          + ("" if s["complete"] else "   [INCOMPLETE]"))


def load_ps(lab):
    d = {}
    with open(R / f"{lab}_per_sample.csv") as fh:
        for r in csv.DictReader(fh):
            if int(r["n_replace"]) in a.nr and (r["dataset"], r["task_id"]) not in EXCL:
                d[(r["dataset"], r["error_type"], int(r["n_replace"]), int(r["T"]), int(r["idx"]))] = (
                    r["task_id"], int(r["test_passed"]))
    return d


if a.pairs:
    print("\nexact-paired McNemar (pooled over cells), A vs B")
for pr in a.pairs:
    A, B = pr.split(":")
    da, db = load_ps(A), load_ps(B)
    res = {}
    for T in range(1, 5):
        keys = [k for k in da if k[3] == T and k in db]
        assert all(da[k][0] == db[k][0] for k in keys), "task_id mismatch -> inputs not identical"
        b = sum(1 for k in keys if da[k][1] == 1 and db[k][1] == 0)
        c = sum(1 for k in keys if da[k][1] == 0 and db[k][1] == 1)
        if not keys:
            continue
        res[f"T{T}"] = {"n": len(keys), "A_only": b, "B_only": c,
                        "micro_diff": (b - c) / len(keys), "mcnemar_p": binom_two_sided(b, c)}
        print(f"  {A} vs {B}  T={T}: n={len(keys)}  A-only={b}  B-only={c}  "
              f"micro diff={(b - c) / len(keys):+.4f}  p={binom_two_sided(b, c):.3g}")
    out["paired"][pr] = res
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
