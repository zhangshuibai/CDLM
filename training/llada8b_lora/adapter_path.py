"""Resolve a LoRA adapter argument to a local directory.

The argument is either a local adapter directory or a Hugging Face model id with an optional
revision, `<owner>/<name>[@<revision>]`, e.g.
`Shuibai12138/LLaDA-8B-CDLM-LoRA-OpenCodeInstruct@<commit>`. A Hub adapter is downloaded
(adapter_config.json and adapter_model.*) with huggingface_hub.snapshot_download.
"""
import os
import re

HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(@[\w./-]+)?$")


def resolve_adapter(spec):
    if spec is None or os.path.isdir(spec):
        return None if spec is None else os.path.abspath(spec)
    if os.path.exists(spec) or not HUB_ID.match(spec):
        raise FileNotFoundError(f"adapter {spec!r} is neither an adapter directory nor a Hub id "
                                "of the form <owner>/<name>[@<revision>]")
    from huggingface_hub import snapshot_download
    repo_id, _, revision = spec.partition("@")
    path = snapshot_download(repo_id, revision=revision or None,
                             allow_patterns=["adapter_config.json", "adapter_model.*"])
    print(f"adapter {spec} -> {path}", flush=True)
    return path
