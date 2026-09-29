"""Resolve a LoRA adapter argument to a local directory.

The argument is either a local adapter directory or a Hugging Face model id with an optional
subfolder and revision, `<owner>/<name>[/<subfolder>][@<revision>]`, e.g.
`Shuibai12138/LLaDA-8B-CDLM-LoRA@<commit>` or `Shuibai12138/LLaDA-8B-CDLM-LoRA/step200@<commit>`.
A Hub adapter is downloaded (adapter_config.json and adapter_model.*) with
huggingface_hub.snapshot_download.
"""
import os
import re

HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(/[\w.-]+)*(@[\w./-]+)?$")


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
