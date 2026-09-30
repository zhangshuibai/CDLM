"""
Thin non-invasive wrapper around the repo's `refine_code.py` that can attach a
LoRA adapter to the base model.

It monkey-patches `utils.load_model_and_tokenizer` (which `refine_code.refine_main`
calls) so that the base model is loaded from a local snapshot at a pinned revision
(`--model_revision`, by default adapter_path.BASE_REVISION for GSAI-ML/LLaDA-8B-Base) and,
when --lora_adapter is given, the adapter is loaded and merged into the bf16 base weights
before refinement. Nothing in the repo is modified; the whole refinement / remasking /
history-dumping code path is byte-identical to the baseline, and refine_code still receives
`--model_name` unchanged.

Usage is identical to refine_code.py plus `--lora_adapter <path>`, where <path> is a local
adapter directory or a Hub id `<owner>/<name>[/<subfolder>][@<revision>]` (see adapter_path.py). Output paths are
built relative to the repo root (the process chdirs there), so pass
`--initial_results_file` relative to the repo root, e.g. `buggy_datasets/...`.
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import utils  # noqa: E402

_orig_loader = utils.load_model_and_tokenizer


def make_loader(model_name_arg, revision, adapter_path):
    """Loader that reads `model_name_arg` from its local snapshot at the pinned revision and merges
    `adapter_path` if given. The snapshot is resolved only when refine_code actually loads the
    model, so cells skipped by --skip_existing need neither the Hub nor the cached base model.
    The snapshot path keeps the model name in it, so the loader's LLaDA mask-token routing is
    unchanged."""
    def loader(model_name, device=None, local_rank=None):
        from adapter_path import pinned_base
        path = pinned_base(model_name, revision) if model_name == model_name_arg else model_name
        model, tokenizer, pad_id, mask_id = _orig_loader(path, device=device, local_rank=local_rank)
        if adapter_path is None:
            return model, tokenizer, pad_id, mask_id
        from peft import PeftModel
        if device is None:
            device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        model = PeftModel.from_pretrained(model, adapter_path, dtype=torch.bfloat16)
        model = model.merge_and_unload().to(device).eval()
        print(f"[LoRA] merged adapter {adapter_path}", flush=True)
        return model, tokenizer, pad_id, mask_id
    return loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_adapter", type=str, default=None)
    p.add_argument("--initial_results_file", type=str, required=True)
    p.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--model_revision", type=str, default=None,
                   help="Hub revision of --model_name; default: the pinned revision for "
                        "GSAI-ML/LLaDA-8B-Base (adapter_path.BASE_REVISION), none otherwise")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--refined_steps", type=int, default=2)
    p.add_argument("--algorithm", type=str, default="self_conf-remask:vanilla")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--refine_setting", type=str, default="remove_all")
    p.add_argument("--confidence_threshold", type=float, default=None)
    p.add_argument("--output_prefix", type=str, default="correction")
    p.add_argument("--mad_k", type=float, default=2.5)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    from adapter_path import resolve_adapter
    utils.load_model_and_tokenizer = make_loader(
        args.model_name, args.model_revision,
        resolve_adapter(args.lora_adapter) if args.lora_adapter else None)

    os.chdir(REPO)
    import refine_code
    refine_code.load_model_and_tokenizer = utils.load_model_and_tokenizer

    refine_code.refine_main(
        initial_results_file=args.initial_results_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        refined_steps=args.refined_steps,
        algorithm=args.algorithm,
        temperature=args.temperature,
        refine_setting=args.refine_setting,
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
        mad_k=args.mad_k,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
