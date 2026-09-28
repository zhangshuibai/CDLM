# Export a veomni DCP checkpoint (--train.ckpt_manager=dcp) as a HuggingFace model directory.
#
# Adapted from scripts/mereg_dcp_to_hf.py of Open-dLLM / VeOmni (Apache-2.0), which targets
# bytecheckpoint. The conversion is the one tasks/train_torch.py runs after each save
# (ckpt_to_state_dict + save_model_weights in bfloat16), so the weights match the hf_ckpt*
# directories written during training. Config and tokenizer files are copied verbatim from the
# run's model_assets/ directory, which is what the in-training export writes as well.
import argparse
import os
import shutil

import torch

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_weights
from veomni.utils import helper


logger = helper.create_logger(__name__)


def read_global_step(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "extra_state", "extra_state_rank_0.pt")
    if not os.path.isfile(path):
        return None
    extra_state = torch.load(path, map_location="cpu", weights_only=False)
    return extra_state.get("global_step")


def main():
    parser = argparse.ArgumentParser(description="Convert a veomni DCP checkpoint to HuggingFace format.")
    parser.add_argument("--ckpt-dir", required=True, help="checkpoint directory, e.g. <run>/checkpoints/global_step_2000")
    parser.add_argument("--model-assets-dir", required=True, help="<run>/model_assets (config and tokenizer files)")
    parser.add_argument("--save-dir", required=True, help="output HuggingFace directory")
    parser.add_argument("--expect-step", type=int, default=None, help="fail if the checkpoint is from another step")
    args = parser.parse_args()

    model_dir = os.path.join(args.ckpt_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No dcp model shards found in {model_dir}")
    if not os.path.isfile(os.path.join(args.model_assets_dir, "config.json")):
        raise FileNotFoundError(f"No config.json found in {args.model_assets_dir}")

    global_step = read_global_step(args.ckpt_dir)
    logger.info(f"Checkpoint {args.ckpt_dir}: global_step={global_step}")
    if args.expect_step is not None and global_step is not None and global_step != args.expect_step:
        raise ValueError(f"{args.ckpt_dir} holds global_step {global_step}, expected {args.expect_step}")

    state_dict = ckpt_to_state_dict(
        save_checkpoint_path=args.ckpt_dir,
        output_dir=args.save_dir,
        ckpt_manager="dcp",
    )
    save_model_weights(args.save_dir, state_dict)

    for name in sorted(os.listdir(args.model_assets_dir)):
        src = os.path.join(args.model_assets_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.save_dir, name))
    logger.info(f"HuggingFace checkpoint saved at {args.save_dir}")


if __name__ == "__main__":
    main()
