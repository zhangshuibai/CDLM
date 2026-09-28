"""Aggregate the G6 (g6v_*) CRB sweep written by run_crb_cell.sh: raw and de-fenced Pass@1 tables."""
import argparse
import glob
import json
import os

DATASETS = ["human-eval", "human-eval+", "mbpp", "mbpp+"]
ERRS = ["operator", "var", "literal"]
LABELS = ["base", "mdlm", "cdlm"]
PRETTY = {"base": "LLaDA-8B-Base", "mdlm": "MDLM (LoRA, 2k steps)",
          "cdlm": "CDLM (LoRA, 2k steps)"}
COLS = [(d, e) for d in DATASETS for e in ERRS]
SHORT = {"human-eval": "he", "human-eval+": "he+", "mbpp": "mbpp", "mbpp+": "mbpp+"}
ESHORT = {"operator": "op", "var": "var", "literal": "lit"}


def load(repo, summary_name, n_replace=1):
    res = {lab: {} for lab in LABELS}
    for lab in LABELS:
        pat = os.path.join(repo, f"g6v_{lab}_results", "refined_steps*", "remove_all",
                           "self_conf-remask_vanilla_ct090_t00", "buggy_datasets", "*",
                           "evaluated", "*", summary_name)
        for f in glob.glob(pat):
            try:
                entries = json.load(open(f))
            except Exception:
                continue
            for md in (entries if isinstance(entries, list) else [entries]):
                if int(md.get("n_replace", 1)) != n_replace:
                    continue
                if md.get("pass_at_1") is None:
                    continue
                res[lab][(md["dataset"], md["error_type"],
                          int(md["refined_steps"]))] = float(md["pass_at_1"])
    return res


def load_initial(repo, n_replace=1):
    out = {}
    for ds, et in COLS:
        f = os.path.join(repo, "buggy_datasets", ds, "evaluated",
                         f"LLaDA-8B-Base_{et}_2_wrong_{n_replace}_evaluated.jsonl")
        if os.path.exists(f):
            rows = [json.loads(l) for l in open(f)]
            if rows:
                out[(ds, et)] = sum(float(r.get("pass@1", 0)) for r in rows) / len(rows)
    return out


def fmt(v):
    return "--" if v is None else f"{100*v:.1f}"


def table(res, init, st, T, title, L):
    L.append(f"**{title} — refined_steps={st} (paper T={T})**\n")
    L.append("| model | " + " | ".join(f"{SHORT[d]}/{ESHORT[e]}" for d, e in COLS) +
             " | **mean** |")
    L.append("|" + "---|" * (len(COLS) + 2))
    iv = [init.get(c) for c in COLS]
    io = [v for v in iv if v is not None]
    L.append("| buggy input (no refinement) | " + " | ".join(fmt(v) for v in iv) +
             f" | {fmt(sum(io)/len(io) if io else None)} |")
    for lab in LABELS:
        vals = [res[lab].get((d, e, st)) for d, e in COLS]
        ok = [v for v in vals if v is not None]
        mean = sum(ok) / len(ok) if ok else None
        note = "" if len(ok) == len(COLS) else f" _(partial {len(ok)}/{len(COLS)})_"
        L.append(f"| {PRETTY[lab]}{note} | " + " | ".join(fmt(v) for v in vals) +
                 f" | **{fmt(mean)}** |")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--n_replace", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init = load_initial(args.repo, args.n_replace)
    raw = load(args.repo, "pass_at_1_summary.json", args.n_replace)
    dfc = load(args.repo, "pass_at_1_summary_defenced.json", args.n_replace)

    L = [f"## Pass@1 (%), CRB n_replace={args.n_replace}, tau=0.9, remove_all, "
         f"self_conf-remask:vanilla, batch_size=1\n"]
    for st, T in ((2, 1), (5, 4)):
        table(raw, init, st, T, "RAW (`--no_postprocess`, paper protocol)", L)
    for st, T in ((2, 1), (5, 4)):
        table(dfc, init, st, T,
              "DE-FENCED (trailing ``` stripped, identical rule for all arms)", L)

    txt = "\n".join(L)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(txt + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
