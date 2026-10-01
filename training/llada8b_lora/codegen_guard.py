"""
From-scratch code generation of LLaDA-8B-Base with an optional LoRA adapter, on the HumanEval+ and MBPP+
problems (a guard metric next to CRB: does post-training keep the model's ability to write programs?).

Benchmarks, prompts, stop sequences, the code-extraction filter and the metric are those of the 0.5B
code-generation evaluation (`evaluation/codegen/`): the vendored lm-evaluation-harness tasks `humaneval_plus`
and `mbpp_plus`, loaded at the pinned dataset revisions and checked against the pinned input digests of
`evaluation/codegen/inputs/INPUTS.json`, the tasks' `create_test` filter, and the HF `code_eval` metric pinned
by `evaluation/codegen/MANIFEST.md5`. Nothing of that path is re-implemented here. What the tasks test:
- `humaneval_plus` runs the EvalPlus HumanEval+ tests (the dataset's `test` field).
- `mbpp_plus` runs only the first three original MBPP asserts of each of the 378 EvalPlus problems, which are
  the asserts printed in the prompt (`doc_to_target` of the harness's mbpp.yaml). This is the primary MBPP+
  score, as in the 0.5B evaluation. The EvalPlus plus tests (`test_imports` + `test`) are scored as well and
  reported as `pass@1_plus_tests`.

Additions to that path, all applied identically to every model:
- The humaneval_plus filter passes every response through the harness's AST sanitizer
  (`lm_eval/tasks/humaneval/sanitize.py`: longest syntactically valid snippet, then only imports, classes,
  functions with a return and assignments); the mbpp_plus filter only deletes code fences. The guard also
  applies the harness's identical `lm_eval/tasks/mbpp/sanitize.py` (no entry point) after the mbpp_plus filter.
  The score of the task filter alone (the 0.5B protocol) is reported as `pass@1_task_filter`.
- code_eval runs with a 20 s timeout per problem instead of its 3 s default: under 3 s some HumanEval+ test
  suites time out even for the canonical solution, depending on machine load.

Generation uses the repository's own sampler (`llada_sample.llada_sample`), the decoder of the paper's LLaDA
self-revision generations: the prompt (tokenised without special tokens, as in the 8B fine-tuning data and the
CRB refinement), then 128 mask tokens, decoded in 128 steps with `self_conf-remask:vanilla` (at every step all
masked positions are predicted greedily, then the lowest-confidence fraction of the 128 positions given by the
linear schedule is masked again), temperature 0. The generated tokens are cut at the first end-of-text token
(LLaDA-8B-Base writes the program, emits <|endoftext|> and starts an unrelated document in the remaining
positions), decoded without special tokens, and cut at each of the task's `until` stop sequences in turn, as
the 0.5B generator does. One greedy sample per problem, batch size 1, so a fixed model gives the same programs
on any GPU count or sharding; the metric is pass@1.

    python codegen_guard.py run   --task humaneval_plus --out <dir> [--adapter A] [--shard k --num_shards n]
    python codegen_guard.py merge --out <dir> --num_shards n [--tasks humaneval_plus mbpp_plus]

`run_codegen_guard.sh` drives both. Generated code is executed: run it in an isolated environment.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

TASKS = ["humaneval_plus", "mbpp_plus"]
GEN_PROTOCOL = {
    "sampler": "llada_sample.llada_sample",
    "algorithm": "self_conf-remask:vanilla",
    "remask_scheduler": "linear",
    "max_new_tokens": 128,
    "steps": 128,
    "temperature": 0.0,
    "samples_per_problem": 1,
    "batch_size": 1,
    "prompt_special_tokens": False,
    "cut_at_first_eos": True,
}
SCORE_PROTOCOL = {
    "metric": "HF code_eval pass@1 (evaluation/codegen/MANIFEST.md5 module), mean over problems",
    "code_eval_timeout_s": 20.0,
    "extra_filter": {"mbpp_plus": "lm_eval.tasks.mbpp.sanitize.sanitize(response) after the task filter"},
    "mbpp_plus_primary_tests": "the task's doc_to_target: test_list[0:3], the asserts shown in the prompt",
    "mbpp_plus_plus_tests": "test_imports + test (EvalPlus), reported as pass@1_plus_tests",
}
CODE_FILES = ["training/llada8b_lora/codegen_guard.py", "training/llada8b_lora/adapter_path.py", "llada_sample.py"]


def _load_codegen_eval():
    """The 0.5B evaluation module, for its pinned task loading and input / metric checks."""
    spec = importlib.util.spec_from_file_location(
        "codegen_eval", os.path.join(REPO, "evaluation", "codegen", "codegen_eval.py"))
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    ce._setup_paths()   # repository lm-eval first on sys.path; HF_ALLOW_CODE_EVAL; HF_SCRIPTS_VERSION=v0.4.0
    return ce


def _md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def code_digests():
    out = {}
    for rel in CODE_FILES:
        with open(os.path.join(REPO, rel), "rb") as f:
            out[rel] = hashlib.md5(f.read()).hexdigest()
    return out


def adapter_digest(adapter_dir):
    """sha256 prefix of adapter_config.json + adapter_model.*, as run_crb_cell.sh stamps adapters."""
    if adapter_dir is None:
        return "NONE"
    h = hashlib.sha256()
    for path in [os.path.join(adapter_dir, "adapter_config.json")] + sorted(glob.glob(os.path.join(adapter_dir, "adapter_model.*"))):
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    return "sha256:" + h.hexdigest()[:16]


def cmd_run(a):
    import torch
    ce = _load_codegen_eval()
    import lm_eval
    assert os.path.realpath(lm_eval.__file__).startswith(os.path.realpath(os.path.join(REPO, "Open-dLLM")) + os.sep), \
        lm_eval.__file__
    task = ce.load_task(a.task)
    inputs = ce.check_inputs(task, a.task)
    docs = list(task.eval_docs)
    idx = list(range(a.shard, len(docs), a.num_shards))
    if a.limit is not None:
        idx = idx[: a.limit]

    from transformers import AutoModel, AutoTokenizer
    from adapter_path import base_revision, resolve_adapter
    from llada_sample import llada_sample
    revision = base_revision(a.model_path, a.model_revision)
    adapter_dir = resolve_adapter(a.adapter) if a.adapter else None
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, revision=revision)
    model = AutoModel.from_pretrained(a.model_path, trust_remote_code=True, dtype=torch.bfloat16,
                                      revision=revision)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, dtype=torch.bfloat16).merge_and_unload()
    device = torch.device("cuda")
    model = model.to(device).eval()
    mask_id, eos = 126336, tok.eos_token_id
    assert tok.convert_ids_to_tokens(mask_id) == "<|mdm_mask|>" and eos is not None

    until = task.config.generation_kwargs["until"]
    os.makedirs(os.path.join(a.out, a.task), exist_ok=True)
    path = os.path.join(a.out, a.task, f"shard{a.shard}of{a.num_shards}.jsonl")
    tmp = path + ".tmp"
    t1 = time.time()
    with open(tmp, "w") as f, torch.no_grad():
        for i in idx:
            prompt = task.doc_to_text(docs[i])
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            n_new = GEN_PROTOCOL["max_new_tokens"]
            x = torch.tensor([ids + [mask_id] * n_new], device=device)
            fix = torch.zeros_like(x, dtype=torch.bool)
            fix[:, : len(ids)] = True
            out = llada_sample(model=model, input_ids=x, fix_mask=fix, mask_id=mask_id,
                               steps=GEN_PROTOCOL["steps"], algorithm=GEN_PROTOCOL["algorithm"],
                               temperature=GEN_PROTOCOL["temperature"],
                               remask_scheduler=GEN_PROTOCOL["remask_scheduler"], model_name=a.model_path)
            gen = out["sequences"][0, len(ids):].tolist()
            eos_index = gen.index(eos) if eos in gen else None
            raw = tok.decode(gen if eos_index is None else gen[:eos_index], skip_special_tokens=True)
            resp = raw
            for stop in until:              # Open-dLLM eval.py generate_until
                if stop in resp:
                    resp = resp.split(stop)[0]
            f.write(json.dumps({"doc_id": i, "prompt_md5": _md5(prompt), "gen_ids": gen, "eos_index": eos_index,
                                "n_mask_left": gen.count(mask_id), "raw": raw, "resp": resp},
                               ensure_ascii=False) + "\n")
            f.flush()
    meta = {"task": a.task, "shard": a.shard, "num_shards": a.num_shards, "limit": a.limit, "n_docs": len(idx),
            "model_path": a.model_path, "model_revision": revision, "adapter": a.adapter_spec or a.adapter,
            "adapter_dir": adapter_dir, "adapter_digest": adapter_digest(adapter_dir),
            "gen_protocol": GEN_PROTOCOL, "code": code_digests(), "inputs": inputs, "t_load_s": round(t1 - t0, 1),
            "t_gen_s": round(time.time() - t1, 1), "cuda_device": torch.cuda.get_device_name(0),
            "torch": torch.__version__}
    with open(path + ".meta.json", "w") as f:       # meta first: a .jsonl without its meta is never final
        json.dump(meta, f, indent=1)
    os.replace(tmp, path)
    print("[guard shard done]", json.dumps({k: meta[k] for k in ("task", "shard", "n_docs", "t_gen_s")}), flush=True)


def _score(code_eval, refs, preds, keys, timeout):
    scores, details = code_eval.compute(references=refs, predictions=preds, k=[1], timeout=timeout)
    res = {k: details[j][0][1] for j, k in enumerate(keys)}      # details are keyed by submission index
    return float(scores["pass@1"]), {k: bool(r["passed"]) for k, r in res.items()}, {k: r["result"] for k, r in res.items()}


def cmd_merge(a):
    ce = _load_codegen_eval()
    path = os.path.join(a.out, "summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {"tasks": {}}
    summary.update({"score_protocol": SCORE_PROTOCOL})
    timeout = SCORE_PROTOCOL["code_eval_timeout_s"]
    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")
    for name in a.tasks:
        d = os.path.join(a.out, name)
        shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, f"shard*of{a.num_shards}.jsonl")))
        others = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(d, "shard*of*.jsonl"))) - set(shards))
        if others:
            raise SystemExit(f"[guard] {d} also holds shards of another sharding: {others}")
        if len(shards) != a.num_shards:
            raise SystemExit(f"[guard] {d}: {len(shards)} of {a.num_shards} shards present")
        metas = [json.load(open(os.path.join(d, s + ".meta.json"))) for s in shards]
        for m in metas:
            for k in ("model_path", "model_revision", "adapter_digest", "gen_protocol", "code", "limit"):
                if m[k] != metas[0][k]:
                    raise SystemExit(f"[guard] {name}: shards differ in {k}: {m[k]} vs {metas[0][k]}")
        if metas[0]["gen_protocol"] != GEN_PROTOCOL:
            raise SystemExit(f"[guard] {name}: shards were generated with another protocol: {metas[0]['gen_protocol']}")
        recs = {}
        for s in shards:
            for line in open(os.path.join(d, s)):
                r = json.loads(line)
                assert r["doc_id"] not in recs, (name, r["doc_id"])
                recs[r["doc_id"]] = r
        task = ce.load_task(name)
        inputs = ce.check_inputs(task, name)
        metric = ce.check_metric_module(task)
        docs = list(task.eval_docs)
        complete = metas[0]["limit"] is None and sorted(recs) == list(range(len(docs)))
        keys = sorted(recs)
        # the task's own filter on one response per request, then the pinned code_eval module
        task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
                                system_instruction=None, apply_chat_template=False, fewshot_as_multiturn=False,
                                chat_template=None, tokenizer_name="")
        insts = {inst.doc_id: inst for inst in task.instances}
        assert sorted(insts) == list(range(len(docs)))
        for k in keys:
            assert _md5(task.doc_to_text(docs[k])) == recs[k]["prompt_md5"], (name, k)
            insts[k].resps.append(recs[k]["resp"])
        for k in set(insts) - set(keys):
            insts[k].resps.append("")       # not generated (partial run); not scored
        task.apply_filters()
        task_preds = [insts[k].filtered_resps["create_test"] for k in keys]
        refs = [task.doc_to_target(docs[k]) for k in keys]
        out = {}
        if name == "mbpp_plus":
            from lm_eval.tasks.mbpp.sanitize import sanitize
            preds = [[sanitize(p) for p in ps] for ps in task_preds]
            out["pass@1"], passed, result = _score(code_eval, refs, preds, keys, timeout)
            out["pass@1_task_filter"], passed_tf, result_tf = _score(code_eval, refs, task_preds, keys, timeout)
            plus_refs = ["\n".join(docs[k]["test_imports"]) + "\n" + docs[k]["test"] for k in keys]
            out["pass@1_plus_tests"], passed_plus, result_plus = _score(code_eval, plus_refs, preds, keys, timeout)
        else:
            preds = task_preds
            out["pass@1"], passed, result = _score(code_eval, refs, preds, keys, timeout)
            passed_tf = passed_plus = result_tf = result_plus = None
        with open(os.path.join(d, "samples.jsonl"), "w") as f:
            for j, k in enumerate(keys):
                rec = {kk: v for kk, v in recs[k].items() if kk != "gen_ids"}
                rec.update({"task_filtered": task_preds[j][0], "filtered": preds[j][0],
                            "passed": passed[k], "result": result[k]})
                if passed_tf is not None:
                    rec.update({"passed_task_filter": passed_tf[k], "passed_plus_tests": passed_plus[k],
                                "result_plus_tests": result_plus[k]})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_timeout = sum(1 for k in keys if "timed out" in str(result[k]))
        summary["tasks"][name] = {
            **out, "n_problems": len(keys), "n_passed": sum(passed.values()), "n_timed_out": n_timeout,
            "complete": complete, "num_shards": a.num_shards, "inputs": inputs, "code_eval_md5": metric,
            "model_path": metas[0]["model_path"], "model_revision": metas[0]["model_revision"],
            "adapter": metas[0]["adapter"], "adapter_digest": metas[0]["adapter_digest"],
            "gen_protocol": metas[0]["gen_protocol"], "code": metas[0]["code"],
            "n_no_eos": sum(1 for k in keys if recs[k]["eos_index"] is None),
            "n_mask_left": sum(recs[k]["n_mask_left"] for k in keys),
            "t_gen_s_total": round(sum(m["t_gen_s"] for m in metas), 1),
            "cuda_devices": sorted({m["cuda_device"] for m in metas})}
        extra = "".join(f"; {k} {v:.4f}" for k, v in out.items() if k != "pass@1")
        print(f"[guard] {name}: pass@1 {out['pass@1']:.4f} ({sum(passed.values())}/{len(keys)}){extra}"
              + ("" if complete else " PARTIAL"), flush=True)
    with open(path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=TASKS)
    r.add_argument("--out", required=True)
    r.add_argument("--adapter", default=None, help="local adapter directory or Hub id <owner>/<name>[/<subfolder>][@<revision>]")
    r.add_argument("--adapter_spec", default=None, help="how the adapter was given, recorded in the metadata")
    r.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base")
    r.add_argument("--model_revision", default=None,
                   help="default: the pinned revision for GSAI-ML/LLaDA-8B-Base (adapter_path.BASE_REVISION)")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=None, help="only the first N problems of this shard (spot check)")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--num_shards", type=int, required=True)
    m.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    a = ap.parse_args()
    if a.cmd == "run":
        assert 0 <= a.shard < a.num_shards
        cmd_run(a)
    else:
        cmd_merge(a)


if __name__ == "__main__":
    main()
