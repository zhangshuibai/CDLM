"""Resolve the base model and LoRA adapter arguments of the LLaDA-8B scripts.

Base model: GSAI-ML/LLaDA-8B-Base is loaded at BASE_REVISION unless another revision is given, so
that neither its weights nor its remote modelling code can change under a run. Every reported run
and every released adapter used this revision (Hub `main` since 2025-10-21). A local directory, or
another model id without an explicit revision, is loaded as given.

Adapter: either a local adapter directory or a Hugging Face model id with an optional subfolder and
revision, `<owner>/<name>[/<subfolder>][@<revision>]`, e.g.
`Shuibai12138/LLaDA-8B-CDLM-LoRA@<commit>` or `Shuibai12138/LLaDA-8B-CDLM-LoRA/step200@<commit>`.
A Hub adapter is downloaded (adapter_config.json and adapter_model.*) with
huggingface_hub.snapshot_download.
"""
import os
import re

BASE_MODEL = "GSAI-ML/LLaDA-8B-Base"
BASE_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(/[\w.-]+)*(@[\w./-]+)?$")


def base_revision(model_path, revision=None):
    """Revision to load model_path at: `revision` if given, BASE_REVISION for the LLaDA-8B-Base
    Hub id, otherwise None (a local directory or another model is loaded as given)."""
    if revision:
        return revision
    return BASE_REVISION if model_path == BASE_MODEL else None


def pinned_base(model_path, revision=None):
    """Local snapshot directory of model_path at base_revision(), or model_path itself when it is a
    local directory or no revision applies."""
    rev = base_revision(model_path, revision)
    if rev is None or os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_path, revision=rev)
    print(f"base model {model_path}@{rev} -> {path}", flush=True)
    return path


def resolve_adapter(spec):
    if spec is None or os.path.isdir(spec):
        return None if spec is None else os.path.abspath(spec)
    if os.path.exists(spec) or not HUB_ID.match(spec):
        raise FileNotFoundError(f"adapter {spec!r} is neither an adapter directory nor a Hub id "
                                "of the form <owner>/<name>[/<subfolder>][@<revision>]")
    from huggingface_hub import snapshot_download
    path, _, revision = spec.partition("@")
    parts = path.split("/")
    repo_id, sub = "/".join(parts[:2]), "/".join(parts[2:])
    prefix = f"{sub}/" if sub else ""
    try:
        root = snapshot_download(repo_id, revision=revision or None,
                                 allow_patterns=[f"{prefix}adapter_config.json", f"{prefix}adapter_model.*"])
    except Exception as e:
        raise FileNotFoundError(f"adapter {spec!r} is not a local directory, and downloading "
                                f"{repo_id!r} (revision {revision or 'main'}) from the Hub failed: "
                                f"{type(e).__name__}: {e}") from e
    local = os.path.join(root, sub) if sub else root
    if not os.path.isfile(os.path.join(local, "adapter_config.json")):
        raise FileNotFoundError(f"no adapter_config.json for {spec!r} (looked in {local})")
    print(f"adapter {spec} -> {local}", flush=True)
    return local
