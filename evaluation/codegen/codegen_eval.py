#!/usr/bin/env python
"""Code-generation evaluation of Open-dCoder checkpoints (paper Table 5 top, Appendix Table 9).

HumanEval, HumanEval+, MBPP and MBPP+; Pass@1 and Pass@10 from n = 10 samples per problem
(T = 0.8, top-k 200, 128 new tokens in 128 steps) with the vanilla (alg p2_upgraded) and ReMDM
(alg p2_upgraded_ReMDM) decoders.

Nothing of the generation or scoring path is re-implemented. The script imports the repository's
Open-dLLM/eval/eval_completion/eval.py (CustomCoder), veomni's diffusion_generate and the task
objects of the vendored lm-evaluation-harness 0.4.9.1 (humaneval / humaneval_plus / mbpp / mbpp_plus
yaml, utils, sanitize), and replays lm-eval's simple_evaluate -> evaluate flow (seeds, per-rank
document sharding, `repeats` cloning, padding, filters, process_results, mean aggregation) for a
virtual world size of 4. The paper's numbers come from `accelerate launch --num_processes 4`
(eval_completion/run_eval.sh): rank r saw docs r, r+4, r+8, ... with a freshly seeded CUDA RNG.
Replaying the 4 ranks as independent single-GPU processes reproduces the recorded samples token
for token, whatever the number of physical GPUs.

Sub-commands
  run        evaluate one checkpoint: all shards over a GPU list, merge, summary.json
  preflight  only the checks `run` does before any GPU work (code, packages, model, inputs,
             code_eval); CPU only
  inputs     check (and optionally export) the pinned benchmark inputs; CPU only
  worker     one (task, decoder, virtual rank) shard; launched by `run`
  merge      merge shard files into a samples file and a metrics file
  compare    compare a samples file with a recorded lm-eval samples file or with the paper's runs
  paired     paired per-problem comparison of two samples files (bootstrap CI, sign-flip test)
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ODLLM = os.path.join(REPO, "Open-dLLM")
IMPORT_PATHS = [os.path.join(ODLLM, "eval", "eval_completion"),
                os.path.join(ODLLM, "lm-evaluation-harness"), ODLLM]
# the only task directories indexed; the other lm-eval tasks are never read
TASK_DIRS = [os.path.join(ODLLM, "lm-evaluation-harness", "lm_eval", "tasks", d) for d in ("humaneval", "mbpp")]
MANIFEST = os.path.join(HERE, "MANIFEST.md5")
INPUTS = os.path.join(HERE, "inputs", "INPUTS.json")
REFERENCE = os.path.join(HERE, "reference", "recorded_runs.json")
MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]

# Pinned protocol (values traced to run_eval.sh, eval.py and the task yaml)
PROTOCOL = {
    "max_new_tokens": 128,          # run_eval.sh MAX_NEW_TOKENS
    "steps": 128,                   # run_eval.sh STEPS (one token committed per step, no blocks)
    "temperature": 0.8,             # run_eval.sh TEMPERATURE
    "top_p": "ignored (0.95 passed in model_args but not a CustomCoder kwarg)",
    "top_k": 200,                   # eval.py CustomCoder default
    "alg_temp": 0.5,                # eval.py default, unused by p2_upgraded*
    "algs": {"vanilla": "p2_upgraded", "remdm": "p2_upgraded_ReMDM"},
    "repeats": 10,                  # task yaml: n = 10 samples per problem
    "metrics": "HF code_eval unbiased pass@k, k in {1,10}, mean over problems",
    "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},   # lm-eval CLI defaults
    "virtual_world_size": 4,        # run_eval.sh NUM_PROCESSES
    "batch_size": {"humaneval": 5, "humaneval_plus": 5, "mbpp": 1, "mbpp_plus": 1},
    "model_args_extra": "add_bos_token=true,top_p=0.95 (ignored by CustomCoder, kept verbatim)",
    "dtype": "auto (checkpoint config torch_dtype must be bfloat16)",
}
TASKS = ["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"]
ALG_NAMES = {v: k for k, v in PROTOCOL["algs"].items()}
# versions of the environment that reproduced the recorded runs (evaluation/requirements.txt)
VERIFIED_VERSIONS = {"torch": "2.5.0+cu121", "transformers": "4.54.1", "tokenizers": "0.21.4",
                     "liger-kernel": "0.5.8", "triton": "3.1.0", "accelerate": "1.10.1", "datasets": "3.6.0",
                     "evaluate": "0.4.5", "huggingface-hub": "0.34.4"}


def _setup_paths():
    for p in reversed(IMPORT_PATHS):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # evaluate 0.4.5 would ask for tag v0.4.5 of the code_eval Space (absent) and fall back to its
    # main; v0.4.0 is the verified module (check_metric_module still checks its md5)
    os.environ.setdefault("HF_SCRIPTS_VERSION", "v0.4.0")


def _hash_string(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def doc_digest(resps):
    """sha256 of a doc's raw responses in sample order (same definition as reference/recorded_runs.json)."""
    flat = [s for q in resps for s in q]
    return hashlib.sha256(json.dumps(flat, ensure_ascii=False).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# pins: code manifest, benchmark inputs, metric module, model
# ----------------------------------------------------------------------------
def read_manifest():
    """{path: md5}; paths are relative to the repository root, 'hf-evaluate:' entries are the
    code_eval metric module that evaluate.load('code_eval') fetches from the Hub."""
    out = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, rel = line.split(None, 1)
                out[rel] = h
    return out


def verify_manifest():
    """Every pinned repository file must match MANIFEST.md5 (byte-identical to the verified protocol)."""
    man = read_manifest()
    bad = [rel for rel, h in man.items() if not rel.startswith("hf-evaluate:")
           and (not os.path.isfile(os.path.join(REPO, rel)) or md5_file(os.path.join(REPO, rel)) != h)]
    if bad:
        raise SystemExit(f"[codegen-eval] {len(bad)} pinned files differ from MANIFEST.md5: {bad[:10]}")
    return {"manifest_md5": md5_file(MANIFEST), "files_checked": sum(not r.startswith("hf-evaluate:") for r in man)}


def check_metric_module(task):
    """The code_eval metric the task loaded must be the pinned one (md5 of code_eval.py and execute.py)."""
    man = read_manifest()
    mods = set()
    for fn in task._metric_fn_list.values():
        fn = getattr(fn, "__self__", fn)
        comp = getattr(fn, "__globals__", {}).get("compute_")
        if comp is not None:
            mods.add(sys.modules[type(comp).__module__].__file__)
    assert mods, "code_eval metric module not found"
    out = {}
    for mod in sorted(mods):
        for name in ("code_eval.py", "execute.py"):
            h = md5_file(os.path.join(os.path.dirname(mod), name))
            if h != man["hf-evaluate:" + name]:
                raise SystemExit(f"[codegen-eval] code_eval metric {name} md5 {h} != pinned "
                                 f"{man['hf-evaluate:' + name]} ({mod})")
            out[name] = h
    return out


def check_repo_imports():
    """Every module imported from Open-dLLM/ must be a pinned file."""
    man = read_manifest()
    root = os.path.realpath(ODLLM) + os.sep
    unpinned = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.realpath(f).startswith(root):
            rel = os.path.relpath(os.path.realpath(f), os.path.realpath(REPO))
            if rel not in man:
                unpinned.append(rel)
    if unpinned:
        raise SystemExit(f"[codegen-eval] imported files outside MANIFEST.md5: {sorted(unpinned)[:10]}")


def check_packages():
    """liger-kernel must be importable: without it veomni's Qwen2 runs plain PyTorch layers and the
    samples change. Other differences from the verified versions only warn."""
    import importlib.metadata as md
    import importlib.util
    got = {}
    for p, want in VERIFIED_VERSIONS.items():
        try:
            got[p] = md.version(p)
        except md.PackageNotFoundError:
            got[p] = None
        if got[p] != want:
            print(f"[codegen-eval] WARNING: {p} {got[p]} differs from the verified {want} "
                  f"(evaluation/ENVIRONMENT.md); samples may differ from the recorded runs", flush=True)
    if importlib.util.find_spec("liger_kernel") is None:   # the test veomni uses
        raise SystemExit("[codegen-eval] liger-kernel is not importable: veomni's Qwen2 would silently run plain "
                         "PyTorch layers (pip install liger-kernel==0.5.8; see evaluation/ENVIRONMENT.md)")
    return got


def load_task(name, task_manager=None):
    """lm-eval task object, dataset loaded at the pinned Hub revision."""
    from lm_eval.tasks import TaskManager, get_task_dict
    pin = _load_json(INPUTS)[name]
    tm = task_manager or TaskManager(include_path=TASK_DIRS, include_defaults=False)
    cfg = {"task": name, "dataset_kwargs": {"revision": pin["revision"]}}
    return get_task_dict([cfg], tm)[name]


def input_digests(task, rows=None):
    """doc / prompt hashes as lm-eval logs them, md5 over all docs in doc_id order."""
    from lm_eval.utils import handle_non_serializable
    dh, ph, ex = [], [], hashlib.md5()
    for i, d in enumerate(task.eval_docs):
        prompt = task.doc_to_text(d)
        dh.append(_hash_string(json.dumps(d, indent=2, default=handle_non_serializable, ensure_ascii=False)))
        ph.append(_hash_string(prompt))
        row = {"doc_id": i, "prompt": prompt, "target": str(task.doc_to_target(d)), "doc": d}
        line = json.dumps(row, ensure_ascii=False, default=handle_non_serializable) + "\n"
        ex.update(line.encode("utf-8"))
        if rows is not None:
            rows.append(line)
    return {"n_docs": len(dh), "doc_hash_md5": hashlib.md5("".join(dh).encode()).hexdigest(),
            "prompt_hash_md5": hashlib.md5("".join(ph).encode()).hexdigest(), "inputs_md5": ex.hexdigest()}


def check_inputs(task, name):
    pin = _load_json(INPUTS)[name]
    got = input_digests(task)
    bad = {k: (got[k], pin[k]) for k in got if got[k] != pin[k]}
    if bad:
        raise SystemExit(f"[codegen-eval] {name}: inputs differ from the pinned set (got, pinned): {bad}")
    return got


def resolve_model(spec):
    """Local HF directory, Hub id, or Hub id@revision -> (local directory, provenance)."""
    if os.path.isdir(spec):
        return os.path.abspath(spec), {"local_dir": os.path.abspath(spec)}
    repo_id, _, rev = spec.partition("@")
    if spec.startswith((".", "/", "~")) or repo_id.count("/") != 1:
        raise SystemExit(f"[codegen-eval] {spec}: not a directory and not a Hub id (org/name[@revision])")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, revision=rev or None, allow_patterns=MODEL_FILES)
    commit = os.path.basename(os.path.normpath(path))
    if not rev:
        print(f"[codegen-eval] WARNING: {repo_id} is not pinned; resolved to commit {commit}. "
              f"Pass {repo_id}@<commit> to pin it.", flush=True)
    elif len(rev) == 40 and commit != rev:
        raise SystemExit(f"[codegen-eval] {spec} resolved to {commit}")
    return path, {"hub_id": repo_id, "requested_revision": rev or None, "resolved_commit": commit}


def preflight_model(local):
    """Refuse checkpoints that eval.py would silently mis-evaluate."""
    from transformers import AutoTokenizer
    cfg = _load_json(os.path.join(local, "config.json"))
    # transformers >= 4.56 (the training environment) writes "dtype" instead of "torch_dtype";
    # with dtype=auto, transformers 4.54.1 then infers bf16 from the weights.
    cfg_dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    info = {"architectures": cfg.get("architectures"), "torch_dtype": cfg_dtype,
            "vocab_size": cfg.get("vocab_size")}
    assert cfg.get("architectures") == ["Qwen2ForCausalLM"], cfg.get("architectures")
    assert cfg_dtype == "bfloat16", f"torch_dtype/dtype={cfg_dtype} (the paper models are bf16)"
    assert cfg.get("vocab_size") == 151936, cfg.get("vocab_size")
    st = os.path.join(local, "model.safetensors")
    assert os.path.exists(st), f"no single-file model.safetensors in {local}"
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(st, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h5.update(b)
            h256.update(b)
    info.update(model_safetensors_bytes=os.path.getsize(st), model_safetensors_md5=h5.hexdigest(),
                model_safetensors_sha256=h256.hexdigest())
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    # eval.py would add a fresh '[MASK]' token and resize the embeddings if '<M>' were missing
    assert tok.mask_token == "<M>" and tok.mask_token_id == 151665, (tok.mask_token, tok.mask_token_id)
    assert tok.eos_token == "<|endoftext|>" and tok.pad_token == "<|endoftext|>", (tok.eos_token, tok.pad_token)
    info.update(mask_token=tok.mask_token, mask_token_id=tok.mask_token_id)
    ref = _load_json(REFERENCE)["models"]
    info["published_model"] = next((k for k, v in ref.items()
                                    if v["model_safetensors_sha256"] == info["model_safetensors_sha256"]), None)
    return info


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def cmd_worker(a):
    _setup_paths()
    import numpy as np
    import torch

    # lm-eval simple_evaluate sets the seeds before the LM is constructed (evaluator.py)
    random.seed(PROTOCOL["seeds"]["random"])
    np.random.seed(PROTOCOL["seeds"]["numpy"])
    torch.manual_seed(PROTOCOL["seeds"]["torch"])

    import eval as paper_eval  # Open-dLLM/eval/eval_completion/eval.py, registers 'custom_coder'
    import lm_eval
    import veomni
    for mod, root in ((paper_eval, IMPORT_PATHS[0]), (lm_eval, IMPORT_PATHS[1]), (veomni, ODLLM)):
        assert os.path.realpath(mod.__file__).startswith(os.path.realpath(root) + os.sep), mod.__file__

    bs = PROTOCOL["batch_size"][a.task] if a.batch_size is None else a.batch_size
    assert not any(c in a.model for c in ",="), f"model path must not contain ',' or '=': {a.model}"
    model_args = (f"pretrained={a.model},max_new_tokens={PROTOCOL['max_new_tokens']},steps={PROTOCOL['steps']},"
                  f"add_bos_token=true,temperature={PROTOCOL['temperature']},top_p=0.95,alg={a.alg}")
    t0 = time.time()
    lm = paper_eval.CustomCoder.create_from_arg_string(
        model_args, {"batch_size": bs, "max_batch_size": None, "device": None})
    assert lm.world_size == 1, "worker must run as a single process"

    task = load_task(a.task)
    inputs = check_inputs(task, a.task)
    metric = check_metric_module(task)
    task.set_fewshot_seed(seed=PROTOCOL["seeds"]["fewshot"])
    W, r = a.vworld, a.vrank
    n_docs = len(task.eval_docs)
    task.build_all_requests(limit=None, rank=r, world_size=W, cache_requests=False,
                            rewrite_requests_cache=False, system_instruction=None,
                            apply_chat_template=False, fewshot_as_multiturn=False,
                            chat_template=None, tokenizer_name="")
    reqs = list(task.instances)
    counts = [len(range(k, n_docs, W)) for k in range(W)]
    assert len(reqs) == counts[r], (len(reqs), counts)
    numpad = max(counts) - counts[r] if W > 1 else 0          # evaluator.py: pad ranks to equal length
    if a.max_docs is not None:                                  # prefix of this rank's RNG stream
        reqs = reqs[: a.max_docs]
        numpad = 0
    cloned = []
    for req in reqs:                                            # evaluator.py: clone `repeats` times
        cloned.extend([req] * req.repeats)
    for _ in range(numpad):
        cloned.extend([req] * req.repeats)                      # (sic) lm-eval re-uses the LAST request
    t1 = time.time()
    resps = lm.generate_until(cloned)
    t2 = time.time()
    for x, req in zip(resps, cloned):
        req.resps.append(x)
    task.apply_filters()

    from collections import defaultdict
    from lm_eval.utils import handle_non_serializable
    by_doc = defaultdict(list)
    for inst in reqs:
        by_doc[inst.doc_id].append(inst)
    for v in by_doc.values():
        v.sort(key=lambda x: x.idx)
    filter_keys = list(reqs[0].filtered_resps.keys())
    assert filter_keys == ["create_test"], filter_keys
    fk = filter_keys[0]
    tmp = a.out + ".tmp"
    with open(tmp, "w") as out:
        for doc_id, doc in task.doc_iterator(rank=r, limit=None, world_size=W):
            if doc_id not in by_doc:
                continue
            rq = by_doc[doc_id]
            metrics = task.process_results(doc, [q.filtered_resps[fk] for q in rq])
            rec = {
                "doc_id": doc_id,
                "resps": [q.resps for q in rq],
                "filtered_resps": [q.filtered_resps[fk] for q in rq],
                "n_samples": sum(len(q.resps) for q in rq),
                "doc_hash": _hash_string(json.dumps(rq[0].doc, indent=2, default=handle_non_serializable,
                                                    ensure_ascii=False)),
                "prompt_hash": _hash_string(rq[0].arguments[0]),
                "target_hash": _hash_string(str(task.doc_to_target(doc))),
            }
            rec.update(metrics)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    check_repo_imports()
    t3 = time.time()
    meta = {"task": a.task, "alg": a.alg, "vrank": r, "vworld": W, "batch_size": bs, "model": a.model,
            "max_docs": a.max_docs, "n_docs_shard": len(reqs), "numpad": numpad, "n_generations": len(cloned),
            "inputs": inputs, "code_eval_md5": metric,
            "t_load_s": round(t1 - t0, 1), "t_gen_s": round(t2 - t1, 1), "t_score_s": round(t3 - t2, 1),
            "cuda_device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    os.replace(tmp, a.out)
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("[worker done]", json.dumps(meta), flush=True)


# ----------------------------------------------------------------------------
# merge / reference comparison
# ----------------------------------------------------------------------------
def _mean_stderr(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return mu, sd / math.sqrt(n)


def merge_shards(shards, out_samples, out_metrics, extra=None):
    recs = []
    for s in shards:
        recs.extend(_read_jsonl(s))
    recs.sort(key=lambda x: x["doc_id"])
    ids = [x["doc_id"] for x in recs]
    assert len(ids) == len(set(ids)), "duplicate doc ids across shards"
    with open(out_samples, "w") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    res = {"n_docs": len(recs), "n_samples_total": sum(x["n_samples"] for x in recs),
           "docs_with_n_ne_10": [x["doc_id"] for x in recs if x["n_samples"] != 10],
           "doc_hash_md5": hashlib.md5("".join(x["doc_hash"] for x in recs).encode()).hexdigest(),
           "prompt_hash_md5": hashlib.md5("".join(x["prompt_hash"] for x in recs).encode()).hexdigest(),
           "resps_sha256": hashlib.sha256("".join(doc_digest(x["resps"]) for x in recs).encode()).hexdigest()}
    for k in ("pass@1", "pass@10"):
        if recs and k in recs[0]:
            mu, se = _mean_stderr([x[k] for x in recs])
            res[k] = mu
            res[k + "_stderr"] = se
    if extra:
        res.update(extra)
    with open(out_metrics, "w") as f:
        json.dump(res, f, indent=1)
    return res


def compare_with_reference(recs, model_key, alg, task):
    """Doc-level comparison of samples with the paper's recorded run of a published model."""
    ref = _load_json(REFERENCE)["models"][model_key]["runs"][alg][task]
    dig, passes = ref["doc_digests"], ref["doc_passes"]
    same_resps = same_pass = 0
    diff = []
    for x in recs:
        d = x["doc_id"]
        ok = doc_digest(x["resps"])[:8] == dig[8 * d: 8 * d + 8]
        c = round(x["pass@1"] * x["n_samples"])
        same_resps += ok
        same_pass += (c == passes[d])
        if not ok or c != passes[d]:
            diff.append({"doc_id": d, "resps_identical": ok, "passes": c, "recorded_passes": passes[d]})
    out = {"reference": f"{model_key}/{alg}/{task} ({ref['recorded']})", "docs_compared": len(recs),
           "docs_with_identical_resps": same_resps, "docs_with_same_pass_count": same_pass,
           "first_diffs": diff[:10]}
    if len(recs) == ref["n_docs"]:
        mine = hashlib.sha256("".join(doc_digest(x["resps"]) for x in recs).encode()).hexdigest()
        out["resps_sha256_identical"] = mine == ref["resps_sha256"]
        for k in ("pass@1", "pass@10"):
            out[k] = {"mine": sum(x[k] for x in recs) / len(recs), "recorded": ref[k]}
    out["identical"] = (same_resps == len(recs) and same_pass == len(recs)
                        and out.get("resps_sha256_identical", True))
    return out


def cmd_merge(a):
    print(json.dumps(merge_shards(a.shards, a.out_samples, a.out_metrics), indent=1))


# ----------------------------------------------------------------------------
# run (one model)
# ----------------------------------------------------------------------------
def _env_versions(gpu):
    env = {}
    try:
        from importlib.metadata import version
        for pkg in ("torch", "transformers", "accelerate", "evaluate", "datasets", "huggingface_hub",
                    "tokenizers", "liger_kernel", "flash_attn", "triton", "numpy"):
            try:
                env[pkg] = version(pkg)
            except Exception:
                env[pkg] = None
        env["python"] = sys.version.split()[0]
        env["gpu"] = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=name,driver_version",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        env["repo_commit"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip() or None
        env["repo_describe"] = subprocess.run(["git", "-C", REPO, "describe", "--always", "--dirty", "--abbrev=12"],
                                              capture_output=True, text=True).stdout.strip() or None
        env["hf_scripts_version"] = os.environ.get("HF_SCRIPTS_VERSION")
    except Exception as e:  # bookkeeping never fails the run
        env["error"] = repr(e)
    return env


def preflight(a):
    """The checks done before any GPU work; SystemExit or AssertionError on the first failure."""
    manifest = verify_manifest()
    packages = check_packages()
    local, source = resolve_model(a.model)
    info = preflight_model(local)
    algs = [PROTOCOL["algs"].get(x, x) for x in a.algs.split(",") if x]
    tasks = [t for t in a.tasks.split(",") if t]
    for x in algs:
        assert x in ALG_NAMES, f"unknown decoder {x} (use vanilla, remdm)"
    for t in tasks:
        assert t in TASKS, f"unknown task {t} (use {','.join(TASKS)})"
    # CPU preflight of every task: dataset at its pinned revision, input digests, metric module
    inputs = {}
    for t in tasks:
        task = load_task(t)
        inputs[t] = check_inputs(task, t)
        inputs[t]["code_eval_md5"] = check_metric_module(task)
        del task
    print(f"[preflight] manifest ok ({manifest['files_checked']} files), model {info['model_safetensors_sha256'][:12]} "
          f"({info['published_model'] or 'not a published model'}), inputs ok: {tasks}", flush=True)
    return {"manifest": manifest, "packages": packages, "local": local, "source": source, "info": info,
            "algs": algs, "tasks": tasks, "inputs": inputs}


def cmd_preflight(a):
    _setup_paths()
    p = preflight(a)
    out = {"model": a.model, "model_source": p["source"], "preflight": p["info"], "manifest": p["manifest"],
           "packages": p["packages"], "hf_scripts_version": os.environ.get("HF_SCRIPTS_VERSION"),
           "algs": p["algs"], "inputs": p["inputs"]}
    print(json.dumps(out, indent=1))
    print("[preflight] OK (no GPU work done)", flush=True)


def cmd_run(a):
    _setup_paths()
    t_start = time.time()
    p = preflight(a)
    manifest, local, source, info = p["manifest"], p["local"], p["source"], p["info"]
    algs, tasks, inputs = p["algs"], p["tasks"], p["inputs"]
    W = PROTOCOL["virtual_world_size"]

    os.makedirs(a.out_dir, exist_ok=True)
    shard_dir = os.path.join(a.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    gpus = [g for g in a.gpus.split(",") if g != ""]
    slots = [g for g in gpus for _ in range(a.procs_per_gpu)]
    order = {"mbpp": 0, "mbpp_plus": 1, "humaneval_plus": 2, "humaneval": 3}   # longest jobs first
    jobs = [(alg, t, r) for alg in algs for t in sorted(tasks, key=lambda x: order[x]) for r in range(W)]
    env_base = dict(os.environ)
    # every download happened in the preflight above; workers read the local caches only
    env_base.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_EVALUATE_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1", HF_ALLOW_CODE_EVAL="1", TOKENIZERS_PARALLELISM="false",
                    PYTHONPATH=":".join(IMPORT_PATHS))
    running = {}   # slot -> (Popen, job, log file, start time)
    pending = list(jobs)
    failed = []
    while pending or running:
        for si in range(len(slots)):
            if si in running or not pending:
                continue
            alg, t, r = pending.pop(0)
            shard = os.path.join(shard_dir, f"{t}__{alg}__r{r}of{W}.jsonl")
            if os.path.exists(shard) and os.path.exists(shard + ".meta.json") and not a.force:
                meta = _load_json(shard + ".meta.json")
                if (meta["model"], meta.get("max_docs")) != (local, a.max_docs):
                    raise SystemExit(f"{shard} was made by another model or --max_docs; use a new OUT_DIR or --force")
                print(f"[skip existing] {shard}", flush=True)
                continue
            env = dict(env_base, CUDA_VISIBLE_DEVICES=slots[si])
            cmd = [sys.executable, os.path.abspath(__file__), "worker", "--model", local, "--task", t,
                   "--alg", alg, "--vrank", str(r), "--vworld", str(W), "--out", shard]
            if a.max_docs is not None:
                cmd += ["--max_docs", str(a.max_docs)]
            logf = open(shard + ".log", "w")
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)
            running[si] = (p, (alg, t, r), logf, time.time())
            print(f"[launch] gpu={slots[si]} {t} {alg} r{r}/{W} pid={p.pid}", flush=True)
        time.sleep(5)
        for si in list(running):
            p, job, logf, ts = running[si]
            if p.poll() is not None:
                logf.close()
                print(f"[exit {p.returncode}] {job} {time.time() - ts:.0f}s", flush=True)
                if p.returncode != 0:
                    failed.append(job)
                del running[si]
    if failed:
        raise SystemExit(f"failed shards (see {shard_dir}/*.log): {failed}")

    partial = a.max_docs is not None
    summary = {"model": a.model, "model_source": source, "preflight": info, "manifest": manifest,
               "inputs": inputs, "protocol": PROTOCOL, "max_docs_per_rank": a.max_docs,
               "driver_md5": md5_file(os.path.abspath(__file__)), "env": _env_versions(gpus[0]),
               "gpus": gpus, "procs_per_gpu": a.procs_per_gpu, "results": {}, "reference_check": {}}
    for alg in algs:
        summary["results"][alg] = {}
        for t in tasks:
            shards = [os.path.join(shard_dir, f"{t}__{alg}__r{r}of{W}.jsonl") for r in range(W)]
            metas = [_load_json(s + ".meta.json") for s in shards]
            samples = os.path.join(a.out_dir, f"samples_{t}__{alg}.jsonl")
            res = merge_shards(shards, samples, os.path.join(a.out_dir, f"metrics_{t}__{alg}.json"),
                               extra={"gen_gpu_seconds": round(sum(m["t_gen_s"] for m in metas), 1),
                                      "score_seconds": round(sum(m["t_score_s"] for m in metas), 1)})
            if not partial:
                pin = inputs[t]
                assert (res["n_docs"], res["doc_hash_md5"], res["prompt_hash_md5"]) == (
                    pin["n_docs"], pin["doc_hash_md5"], pin["prompt_hash_md5"]), f"{t}: merged inputs differ from the pins"
            summary["results"][alg][t] = {k: res[k] for k in res if k.startswith("pass@") or k in (
                "n_docs", "n_samples_total", "docs_with_n_ne_10", "doc_hash_md5", "prompt_hash_md5",
                "resps_sha256", "gen_gpu_seconds", "score_seconds")}
            if info["published_model"]:
                chk = compare_with_reference(_read_jsonl(samples), info["published_model"], alg, t)
                summary["reference_check"].setdefault(alg, {})[t] = chk
                print(f"[reference] {alg} {t}: {chk['docs_with_identical_resps']}/{chk['docs_compared']} docs "
                      f"identical to the recorded paper run", flush=True)
        r_ = summary["results"][alg]
        if not partial:
            for k in ("pass@1", "pass@10"):
                if all(t in r_ for t in TASKS):
                    r_[f"macro4_{k}"] = sum(r_[t][k] for t in TASKS) / 4
                if "humaneval" in r_ and "humaneval_plus" in r_:
                    r_[f"he_avg_{k}"] = (r_["humaneval"][k] + r_["humaneval_plus"][k]) / 2
                if "mbpp" in r_ and "mbpp_plus" in r_:
                    r_[f"mbpp_avg_{k}"] = (r_["mbpp"][k] + r_["mbpp_plus"][k]) / 2
    summary["wall_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["results"], indent=1))
    if not a.keep_shards:
        for f in os.listdir(shard_dir):
            if f.endswith(".jsonl"):
                os.remove(os.path.join(shard_dir, f))   # the merged samples file keeps the content
    if summary["reference_check"] and not all(c["identical"] for v in summary["reference_check"].values()
                                              for c in v.values()):
        print("[codegen-eval] WARNING: samples differ from the recorded paper run "
              "(see reference_check in summary.json)", flush=True)


# ----------------------------------------------------------------------------
# inputs / compare / paired
# ----------------------------------------------------------------------------
def cmd_inputs(a):
    _setup_paths()
    out = {}
    for t in [x for x in a.tasks.split(",") if x]:
        task = load_task(t)
        rows = [] if a.export else None
        got = input_digests(task, rows)
        pin = _load_json(INPUTS)[t]
        out[t] = dict(got, match=all(got[k] == pin[k] for k in got), code_eval_md5=check_metric_module(task))
        if a.export:
            os.makedirs(a.export, exist_ok=True)
            with open(os.path.join(a.export, f"{t}.jsonl"), "w") as f:
                f.writelines(rows)
    print(json.dumps(out, indent=1))
    if not all(v["match"] for v in out.values()):
        raise SystemExit("[codegen-eval] inputs differ from inputs/INPUTS.json")


def cmd_compare(a):
    mine = {x["doc_id"]: x for x in _read_jsonl(a.mine)}
    if a.reference:
        assert a.task and a.alg, "--reference needs --task and --alg"
        out = compare_with_reference([mine[d] for d in sorted(mine)], a.reference,
                                     PROTOCOL["algs"].get(a.alg, a.alg), a.task)
    else:
        rec = {x["doc_id"]: x for x in _read_jsonl(a.recorded)}
        common = sorted(set(mine) & set(rec))
        n_doc_same_resps = n_resp = n_resp_same = n_hash_same = n_pass_same = 0
        d1 = d10 = 0.0
        first_diff = None
        for d in common:
            m, r = mine[d], rec[d]
            n_hash_same += (m["doc_hash"] == r["doc_hash"] and m["prompt_hash"] == r["prompt_hash"])
            mr = [s for q in m["resps"] for s in q]
            rr = [s for q in r["resps"] for s in q]
            k = min(len(mr), len(rr))
            same = sum(mr[i] == rr[i] for i in range(k))
            n_resp += k
            n_resp_same += same
            n_doc_same_resps += (mr == rr)
            if mr != rr and first_diff is None:
                i = next(i for i in range(k) if mr[i] != rr[i]) if same < k else k
                first_diff = {"doc_id": d, "sample": i, "mine": mr[i][:200] if i < len(mr) else None,
                              "recorded": rr[i][:200] if i < len(rr) else None}
            n_pass_same += (m.get("pass@1") == r.get("pass@1") and m.get("pass@10") == r.get("pass@10"))
            d1 += abs(m.get("pass@1", 0) - r.get("pass@1", 0))
            d10 += abs(m.get("pass@10", 0) - r.get("pass@10", 0))
        out = {"docs_compared": len(common), "doc+prompt hash identical": n_hash_same,
               "docs with identical raw resps": n_doc_same_resps, "resps compared": n_resp,
               "resps identical": n_resp_same, "docs with identical pass@1 and pass@10": n_pass_same,
               "sum|dpass@1| over docs": d1, "sum|dpass@10| over docs": d10, "first_diff": first_diff}
        if common:
            out["mine_pass@1_on_common"] = sum(mine[d]["pass@1"] for d in common) / len(common)
            out["recorded_pass@1_on_common"] = sum(rec[d]["pass@1"] for d in common) / len(common)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)


def cmd_paired(a):
    """Paired (per-problem) comparison of two samples files (this script's output or recorded lm-eval
    samples): A - B on per-problem pass@k, paired bootstrap 95% CI and two-sided sign-flip permutation p."""
    A = {x["doc_id"]: x for x in _read_jsonl(a.a)}
    B = {x["doc_id"]: x for x in _read_jsonl(a.b)}
    assert set(A) == set(B), "files cover different problems"
    ids = sorted(A)
    for x in ids:
        assert A[x]["doc_hash"] == B[x]["doc_hash"], f"doc {x} differs between files"
    out = {"a": a.a, "b": a.b, "n_problems": len(ids)}
    rng = random.Random(0)
    for k in ("pass@1", "pass@10"):
        d = [A[i][k] - B[i][k] for i in ids]
        n = len(d)
        mu = sum(d) / n
        boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(a.iters))
        ge = sum(1 for _ in range(a.iters) if abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= abs(mu) - 1e-15)
        out[k] = {"A": sum(A[i][k] for i in ids) / n, "B": sum(B[i][k] for i in ids) / n, "A_minus_B": mu,
                  "ci95": [boots[int(0.025 * a.iters)], boots[int(0.975 * a.iters) - 1]],
                  "p_signflip": (ge + 1) / (a.iters + 1),
                  "problems_A_better": sum(x > 0 for x in d), "problems_B_better": sum(x < 0 for x in d)}
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="evaluate one checkpoint")
    r.add_argument("--model", required=True, help="local HF checkpoint dir, Hub id, or Hub id@revision")
    r.add_argument("--out_dir", required=True)
    r.add_argument("--gpus", default="0", help="comma list of physical GPU ids")
    r.add_argument("--procs_per_gpu", type=int, default=4)
    r.add_argument("--algs", default="vanilla,remdm", help="vanilla, remdm or both")
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--max_docs", type=int, default=None,
                   help="spot check: only the first N docs of every virtual rank (a prefix of its RNG stream)")
    r.add_argument("--force", action="store_true", help="regenerate shards that already exist")
    r.add_argument("--keep_shards", action="store_true")
    f = sp.add_parser("preflight", help="only the checks run does before any GPU work (CPU)")
    f.add_argument("--model", required=True, help="local HF checkpoint dir, Hub id, or Hub id@revision")
    f.add_argument("--algs", default="vanilla,remdm", help="vanilla, remdm or both")
    f.add_argument("--tasks", default=",".join(TASKS))
    i = sp.add_parser("inputs", help="check the pinned benchmark inputs (CPU)")
    i.add_argument("--tasks", default=",".join(TASKS))
    i.add_argument("--export", default=None, help="also write <task>.jsonl (doc_id, prompt, target, doc) here")
    w = sp.add_parser("worker", help="one shard (launched by run)")
    w.add_argument("--model", required=True, help="local checkpoint dir")
    w.add_argument("--task", required=True, choices=TASKS)
    w.add_argument("--alg", required=True, choices=list(ALG_NAMES))
    w.add_argument("--vrank", type=int, required=True)
    w.add_argument("--vworld", type=int, default=PROTOCOL["virtual_world_size"])
    w.add_argument("--out", required=True)
    w.add_argument("--batch_size", type=int, default=None, help="override (breaks bitwise equivalence)")
    w.add_argument("--max_docs", type=int, default=None, help="only the first N docs of this shard")
    m = sp.add_parser("merge", help="merge shard files")
    m.add_argument("--shards", nargs="+", required=True)
    m.add_argument("--out_samples", required=True)
    m.add_argument("--out_metrics", required=True)
    c = sp.add_parser("compare", help="compare a samples file with a recorded run")
    c.add_argument("--mine", required=True)
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--recorded", help="recorded lm-eval samples_<task>_*.jsonl")
    g.add_argument("--reference", choices=["base", "mdlm", "cdlm"], help="the paper's recorded run (built-in digests)")
    c.add_argument("--task", choices=TASKS)
    c.add_argument("--alg", help="vanilla, remdm, p2_upgraded or p2_upgraded_ReMDM")
    c.add_argument("--out", default=None)
    pr = sp.add_parser("paired", help="paired comparison of two samples files")
    pr.add_argument("--a", required=True, help="samples jsonl of model A (same task, same decoder)")
    pr.add_argument("--b", required=True, help="samples jsonl of model B")
    pr.add_argument("--iters", type=int, default=10000)
    pr.add_argument("--out", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "preflight": cmd_preflight, "inputs": cmd_inputs, "worker": cmd_worker, "merge": cmd_merge,
     "compare": cmd_compare, "paired": cmd_paired}[a.cmd](a)


if __name__ == "__main__":
    main()
