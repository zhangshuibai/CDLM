"""
Matched-compute MDLM vs CDLM LoRA fine-tuning of LLaDA-8B-Base.

Two arms, byte-identical except for the training objective:

  MDLM arm : mixture_prob=0, noise_token_wt=0, clean_token_wt=0
             -> plain absorbing masked-diffusion loss (LLaDA official 1/t weighting)

  CDLM arm : mixture_prob=0.1, noise_token_wt=0.1, clean_token_wt=0
             -> same absorbing loss, PLUS uniform-replacement corruption on *visible*
                tokens with prob alpha=mixture_prob, supervised with weight lambda_noise.

Corruption semantics follow `sudoku/train.py:add_uniform_absorbing_mixture_noise`
and `veomni/data/data_collator.py:DataCollatorWithPositionIDsMixture_Masking`:
  1. mask each token independently with prob t ~ U(0,1)
  2. for each *remaining visible* token, with prob alpha replace it with a uniformly
     sampled token that is neither the mask token nor the original token
  3. loss is applied ONLY to (masked) + (uniformly-corrupted) positions; never to
     clean visible tokens (unless clean_token_wt > 0).

Launch (4 GPUs):
  torchrun --nproc_per_node=4 train_llada_mixture.py --arm cdlm ...
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

MASK_ID = 126336          # <|mdm_mask|> for LLaDA-8B-Base
PAD_ID = 126081
IGNORE_INDEX = -100

# LoRA on the 7 linear projections *inside transformer blocks only*.
# `model.transformer.ff_out` (the lm_head) and `wte` (embeddings) stay frozen,
# mirroring the 0.5B recipe's `--train.freeze_layers="lm_head,embed_tokens"`.
LORA_TARGET_REGEX = r".*transformer\.blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def split_into_chunks(tokens, max_seq_len):
    for i in range(0, len(tokens), max_seq_len):
        yield tokens[i:i + max_seq_len]


class PackedParquetStream(IterableDataset):
    """Replicates veomni `process_pretrain_example`: tokenize `text`, append EOS,
    split into <=max_seq_len chunks. Shard files round-robin over ranks so that
    the *global* stream is deterministic given (seed, world_size)."""

    def __init__(self, data_dir, tokenizer, max_seq_len, rank, world_size, seed):
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        )
        rng = random.Random(seed)
        rng.shuffle(self.files)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self):
        import pyarrow.parquet as pq

        my_files = self.files[self.rank::self.world_size]
        eos = self.tokenizer.eos_token_id
        while True:  # loop forever; trainer stops at max_steps
            for path in my_files:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=64, columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        if not text:
                            continue
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        ids = ids + [eos]
                        for chunk in split_into_chunks(ids, self.max_seq_len):
                            if len(chunk) < 16:
                                continue
                            yield torch.tensor(chunk, dtype=torch.long)


def collate(seqs, max_seq_len):
    """Right-pad to the longest sequence in the micro-batch."""
    n = len(seqs)
    L = max(len(s) for s in seqs)
    input_ids = torch.full((n, L), PAD_ID, dtype=torch.long)
    valid = torch.zeros((n, L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = s
        valid[i, :len(s)] = True
    return {"input_ids": input_ids, "valid": valid}


# --------------------------------------------------------------------------- #
# forward corruption process
# --------------------------------------------------------------------------- #
def forward_process(input_ids, valid, mixture_prob, vocab_size, gen_abs, gen_mix):
    """
    Returns (noisy_ids, mask_positions, noise_positions, p_mask)

    mask_positions : absorbed (mask-token) positions          -> absorbing loss, 1/t weighted
    noise_positions: uniformly-replaced visible positions     -> noise loss, unweighted mean
    p_mask         : per-token masking probability t (broadcast per sequence)

    Two independent generators so that the *absorbing* noise stream is byte-identical
    between the MDLM and CDLM arms (the mixture draws never perturb it).
    """
    generator = gen_mix
    b, l = input_ids.shape
    device = input_ids.device

    # --- absorbing noise: t ~ U(0,1) per sequence (LLaDA official recipe) ---
    t = torch.rand(b, device=device, generator=gen_abs)
    p_mask = ((1 - 1e-3) * t + 1e-3).unsqueeze(1).expand(b, l)
    mask_positions = (torch.rand((b, l), device=device, generator=gen_abs) < p_mask) & valid

    noisy = input_ids.clone()
    noisy[mask_positions] = MASK_ID

    noise_positions = torch.zeros_like(mask_positions)
    if mixture_prob > 0.0:
        cand = valid & (~mask_positions)
        r = torch.rand((b, l), device=device, generator=generator)
        replace = cand & (r < mixture_prob)
        n_rep = int(replace.sum().item())
        if n_rep > 0:
            orig = input_ids[replace]
            # sample from [0, vocab_size-1) then skip over MASK_ID  -> never == MASK_ID
            new = torch.randint(0, vocab_size - 1, (n_rep,), device=device,
                                dtype=torch.long, generator=generator)
            new = new + (new >= MASK_ID).long()
            # resolve collisions with the original token
            for _ in range(5):
                clash = new == orig
                if not clash.any():
                    break
                k = int(clash.sum().item())
                fresh = torch.randint(0, vocab_size - 1, (k,), device=device,
                                      dtype=torch.long, generator=generator)
                fresh = fresh + (fresh >= MASK_ID).long()
                new[clash] = fresh
            clash = new == orig
            if clash.any():  # deterministic fallback
                new[clash] = (orig[clash] + 1) % vocab_size
                eq = new == MASK_ID
                new[eq] = (new[eq] + 1) % vocab_size
            noisy[replace] = new
            noise_positions = replace

    return noisy, mask_positions, noise_positions, p_mask


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def compute_loss(logits, targets, mask_positions, noise_positions, p_mask,
                 noise_token_wt, clean_token_wt, valid):
    """
    LLaDA is bidirectional -> logits[:, i] predicts token[:, i]; NO shift.

    absorbing (mdm) loss : sum_over_masked( CE / t ) / (B * L)   [LLaDA official]
    noise loss           : mean CE over uniformly-corrupted visible positions
    clean loss           : mean CE over clean visible positions  (unused, wt=0)
    """
    comps = {}
    b, l, _ = logits.shape
    denom = b * l

    if mask_positions.any():
        sel = logits[mask_positions].float()
        ce = F.cross_entropy(sel, targets[mask_positions], reduction="none")
        mdm_loss = (ce / p_mask[mask_positions]).sum() / denom
    else:
        mdm_loss = logits.sum() * 0.0
    loss = mdm_loss
    comps["mdm"] = float(mdm_loss.detach())

    if noise_token_wt and noise_positions.any():
        sel = logits[noise_positions].float()
        ce = F.cross_entropy(sel, targets[noise_positions], reduction="none")
        noise_loss = ce.mean()
        loss = loss + noise_token_wt * noise_loss
        comps["noise"] = float(noise_loss.detach())

    if clean_token_wt:
        clean_positions = valid & (~mask_positions) & (~noise_positions)
        if clean_positions.any():
            sel = logits[clean_positions].float()
            ce = F.cross_entropy(sel, targets[clean_positions], reduction="none")
            clean_loss = ce.mean()
            loss = loss + clean_token_wt * clean_loss
            comps["clean"] = float(clean_loss.detach())

    return loss, comps


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mdlm", "cdlm"])
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--model_revision", default=None,
                    help="Hub revision of --model_path; default: the pinned revision for "
                         "GSAI-ML/LLaDA-8B-Base (adapter_path.BASE_REVISION), none otherwise")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--cosine_horizon", type=int, default=0,
                    help="cosine decay horizon in steps; 0 => use max_steps")
    ap.add_argument("--warmup_ratio", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--global_batch_size", type=int, default=12)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--mixture_prob", type=float, default=None)
    ap.add_argument("--noise_token_wt", type=float, default=None)
    ap.add_argument("--clean_token_wt", type=float, default=0.0)
    ap.add_argument("--save_steps", type=int, default=0, help="0 => only final")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    if args.mixture_prob is None:
        args.mixture_prob = 0.1 if args.arm == "cdlm" else 0.0
    if args.noise_token_wt is None:
        args.noise_token_wt = 0.1 if args.arm == "cdlm" else 0.0

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    def log(msg):
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # identical seeding across arms
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accum = args.global_batch_size // (args.micro_batch_size * world)
    assert accum * args.micro_batch_size * world == args.global_batch_size, \
        "global_batch_size must equal micro*world*accum"

    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from adapter_path import base_revision

    args.model_revision = base_revision(args.model_path, args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              revision=args.model_revision)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      dtype=torch.bfloat16, revision=args.model_revision)
    vocab_size = model.config.vocab_size  # 126464 (== embedding_size)

    # activation checkpointing (LLaDA custom API), set before PEFT wrapping
    try:
        from transformers.models.auto.configuration_auto import AutoConfig  # noqa
        import importlib
        cfgmod = importlib.import_module(type(model.config).__module__)
        strat = cfgmod.ActivationCheckpointingStrategy.whole_layer
        model.model.set_activation_checkpointing(strat)
        log("activation checkpointing: whole_layer")
    except Exception as e:  # pragma: no cover
        log(f"WARNING: could not enable activation checkpointing: {e}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_REGEX,
        task_type=None,
    )
    model = get_peft_model(model, lora_cfg)
    # fp32 LoRA params for stable AdamW at lr 3e-4
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    lora_names = sorted({n for n, p in model.named_parameters() if p.requires_grad})
    log(f"trainable params {n_train/1e6:.2f}M / {n_all/1e9:.3f}B "
        f"({100*n_train/n_all:.3f}%), {len(lora_names)} lora tensors")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                gradient_as_bucket_view=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    horizon = args.cosine_horizon if args.cosine_horizon > 0 else args.max_steps
    warmup = max(1, int(args.warmup_ratio * horizon))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, horizon - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ds = PackedParquetStream(args.data_dir, tokenizer, args.max_seq_len,
                             rank, world, args.seed)
    dl = DataLoader(ds, batch_size=args.micro_batch_size, num_workers=2,
                    collate_fn=lambda b: collate(b, args.max_seq_len),
                    pin_memory=True, prefetch_factor=4)
    it = iter(dl)

    # per-rank generators; separate streams so the absorbing noise is byte-identical
    # between arms regardless of whether mixture draws happen.
    gen_abs = torch.Generator(device=device)
    gen_abs.manual_seed(args.seed * 1000 + rank)
    gen_mix = torch.Generator(device=device)
    gen_mix.manual_seed(args.seed * 1000 + 500000 + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
            json.dump({**vars(args), "world_size": world, "grad_accum": accum,
                       "warmup_steps": warmup, "cosine_horizon_effective": horizon,
                       "lora_target_regex": LORA_TARGET_REGEX,
                       "vocab_size": vocab_size, "mask_id": MASK_ID,
                       "n_trainable": n_train}, f, indent=2)

    log(f"=== arm={args.arm} mixture_prob={args.mixture_prob} "
        f"noise_token_wt={args.noise_token_wt} clean_token_wt={args.clean_token_wt} ===")
    log(f"world={world} micro={args.micro_batch_size} accum={accum} "
        f"global={args.global_batch_size} seq_len={args.max_seq_len} "
        f"steps={args.max_steps} horizon={horizon} warmup={warmup}")

    hist = []
    model.train()
    t0 = time.time()
    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss, step_comps = 0.0, {}
        for micro in range(accum):
            batch = next(it)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            noisy, mpos, npos, p_mask = forward_process(
                input_ids, valid, args.mixture_prob, vocab_size, gen_abs, gen_mix)

            ctx = model.no_sync() if micro < accum - 1 else torch.enable_grad()
            with ctx:
                out = model(input_ids=noisy)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss, comps = compute_loss(
                    logits, input_ids, mpos, npos, p_mask,
                    args.noise_token_wt, args.clean_token_wt, valid)
                (loss / accum).backward()
            step_loss += float(loss.detach()) / accum
            for k, v in comps.items():
                step_comps[k] = step_comps.get(k, 0.0) + v / accum
            del logits, out

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            stats = torch.tensor([step_loss, float(gnorm)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats /= world
            cs = " ".join(f"{k}:{v:.3f}" for k, v in sorted(step_comps.items()))
            log(f"step {step+1}/{args.max_steps} loss {stats[0]:.4f} {cs} "
                f"gnorm {stats[1]:.2f} lr {scheduler.get_last_lr()[0]:.2e} "
                f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G "
                f"{(time.time()-t0)/(step+1):.2f}s/step")
            if rank == 0:
                hist.append({"step": step + 1, "loss": float(stats[0]),
                             "lr": scheduler.get_last_lr()[0], **step_comps})

        if args.save_steps and (step + 1) % args.save_steps == 0:
            if rank == 0:
                d = os.path.join(args.output_dir, f"step{step+1}")
                model.module.save_pretrained(d)
                log(f"saved {d}")
            dist.barrier()

    if rank == 0:
        model.module.save_pretrained(os.path.join(args.output_dir, "final"))
        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(hist, f, indent=2)
        log(f"DONE in {(time.time()-t0)/60:.1f} min -> {args.output_dir}/final")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
