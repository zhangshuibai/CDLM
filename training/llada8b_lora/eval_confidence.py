"""
Error-localization metrics for LLaDA-8B on the Code Repair Benchmark (CRB).

Reproduces exactly the input construction used by `refine_code.py` with
`--refine_setting remove_all`:

    input_ids = tok(function_head + "\n") ++ tok(buggy_body)      (no special tokens)
    fixed context = the function_head part; variable = the buggy_body part

Confidence of the *currently visible* token, exactly as `llada_sample.py:602-606`:

    probs    = softmax(logits.float(), -1)
    prob_all = gather(probs, -1, x.unsqueeze(-1))

LLaDA is bidirectional (`is_causal=False` hard-coded in modeling_llada.py), so
logits[:, i] predicts token[:, i] and there is NO logit shift (cf. llada_sample.py:560,
which shifts only for open-dcoder / Dream-v0 / Fast-dLLM).

Reported:
  * confidence gap   = mean(conf over clean body positions) - mean(conf over error positions)
  * Top-K hit rate   = fraction of samples where >=1 true error position is among the
                       K lowest-confidence body positions (K = 1..6)
Both for the gather-based confidence (primary, matches the sampler) and the
max-over-vocab confidence (secondary, matches the prose in the paper).
"""

import argparse
import json
import os

import numpy as np
import torch

MASK_ID = 126336
MAX_K = 6


def load_model(model_path, adapter, device, revision=None):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      dtype=torch.bfloat16, revision=revision)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, dtype=torch.bfloat16)
        model = model.merge_and_unload()
        print(f"merged LoRA adapter: {adapter}", flush=True)
    return model.to(device).eval(), tok


@torch.no_grad()
def eval_file(model, tok, path, device, limit=None):
    rows = [json.loads(l) for l in open(path)]
    if limit:
        rows = rows[:limit]

    recs = []
    for item in rows:
        err_pos = item.get("error_positions") or []
        if not err_pos:
            continue
        head = f"{item['function_head']}\n"
        ctx = tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        body = tok(item["buggy_body"], return_tensors="pt",
                   add_special_tokens=False)["input_ids"][0]
        if max(err_pos) >= len(body):
            continue
        x = torch.cat([ctx, body]).unsqueeze(0).to(device)

        logits = model(input_ids=x).logits          # no shift: LLaDA is bidirectional
        probs = torch.softmax(logits.float(), dim=-1)
        gather_conf = torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1)[0]
        max_conf = probs.max(dim=-1).values[0]

        off = len(ctx)
        g = gather_conf[off:off + len(body)].cpu().numpy()
        m = max_conf[off:off + len(body)].cpu().numpy()
        err = np.zeros(len(body), dtype=bool)
        err[np.asarray(err_pos, dtype=int)] = True

        rec = {"task_id": item.get("task_id"), "n_body": int(len(body)),
               "n_err": int(err.sum())}
        for tag, c in (("gather", g), ("max", m)):
            rec[f"{tag}_clean"] = float(c[~err].mean()) if (~err).any() else float("nan")
            rec[f"{tag}_err"] = float(c[err].mean())
            rec[f"{tag}_gap"] = rec[f"{tag}_clean"] - rec[f"{tag}_err"]
            order = np.argsort(c, kind="stable")           # ascending confidence
            for k in range(1, MAX_K + 1):
                topk = set(order[:k].tolist())
                rec[f"{tag}_hit@{k}"] = int(any(p in topk for p in np.where(err)[0]))
        recs.append(rec)
    return recs


def summarize(recs):
    out = {"n_samples": len(recs)}
    if not recs:
        return out
    for tag in ("gather", "max"):
        for f in ("clean", "err", "gap"):
            out[f"{tag}_{f}"] = float(np.mean([r[f"{tag}_{f}"] for r in recs]))
        for k in range(1, MAX_K + 1):
            out[f"{tag}_hit@{k}"] = float(np.mean([r[f"{tag}_hit@{k}"] for r in recs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--model_revision", default=None,
                    help="Hub revision of --model_path; default: the pinned revision for "
                         "GSAI-ML/LLaDA-8B-Base (adapter_path.BASE_REVISION), none otherwise")
    ap.add_argument("--adapter", default=None,
                    help="local adapter directory or Hub id <owner>/<name>[/<subfolder>][@<revision>]")
    ap.add_argument("--label", required=True)
    ap.add_argument("--data_root", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "buggy_datasets")))
    ap.add_argument("--file_prefix", default="LLaDA-8B-Base")
    ap.add_argument("--datasets", nargs="+", default=["human-eval"])
    ap.add_argument("--error_types", nargs="+", default=["operator", "var", "literal"])
    ap.add_argument("--n_replace", nargs="+", type=int, default=[1])
    ap.add_argument("--data_num", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    from adapter_path import base_revision, resolve_adapter
    revision = base_revision(args.model_path, args.model_revision)
    adapter_dir = resolve_adapter(args.adapter)
    model, tok = load_model(args.model_path, adapter_dir, device, revision)

    results = {"label": args.label, "model_path": args.model_path, "model_revision": revision,
               "adapter": args.adapter, "adapter_dir": adapter_dir, "per_file": {}}
    pooled = []
    for ds in args.datasets:
        for et in args.error_types:
            for nr in args.n_replace:
                p = os.path.join(args.data_root, ds,
                                 f"{args.file_prefix}_{et}_{args.data_num}_wrong_{nr}.jsonl")
                if not os.path.exists(p):
                    print(f"MISSING {p}", flush=True)
                    continue
                recs = eval_file(model, tok, p, device, args.limit)
                key = f"{ds}/{et}/n{nr}"
                results["per_file"][key] = summarize(recs)
                pooled.extend(recs)
                s = results["per_file"][key]
                print(f"{args.label} {key}: n={s['n_samples']} "
                      f"gap={s['gather_gap']:.4f} (clean {s['gather_clean']:.4f} / "
                      f"err {s['gather_err']:.4f})  hit@1={s['gather_hit@1']:.3f} "
                      f"hit@5={s['gather_hit@5']:.3f}", flush=True)
    results["pooled"] = summarize(pooled)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    s = results["pooled"]
    print(f"\n=== POOLED [{args.label}] n={s['n_samples']} ===")
    print(f"  gather: gap={s['gather_gap']:.4f} clean={s['gather_clean']:.4f} "
          f"err={s['gather_err']:.4f}")
    print("  gather hit@K: " + " ".join(f"K{k}={s[f'gather_hit@{k}']:.3f}"
                                        for k in range(1, MAX_K + 1)))
    print(f"  max   : gap={s['max_gap']:.4f} clean={s['max_clean']:.4f} "
          f"err={s['max_err']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
