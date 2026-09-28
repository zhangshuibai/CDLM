#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Sudoku Diffusion Language Model
Implements discrete diffusion training with dynamic masking
"""

import json
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from model import create_sudoku_dit


class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles and solutions."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        
        # Load data
        self.solutions = np.load(self.data_dir / f'{split}_solutions.npy')
        
        # Load metadata
        with open(self.data_dir / f'{split}_metadata.json') as f:
            self.metadata = json.load(f)
        
        self.mask_id = 0  # MASK token (0 = empty cell in sudoku)
        print(f"Loaded {len(self.solutions)} {split} samples")
    
    def __len__(self):
        return len(self.solutions)
    
    def __getitem__(self, idx):
        # Only use solutions for training (diffusion learns to denoise)
        solution = torch.from_numpy(self.solutions[idx]).long()
        return {'solution': solution}


def add_absorbing_noise(tokens, mask_id, mask_ratio_min=0.2, mask_ratio_max=0.9):
    """
    Apply absorbing noise (random masking) for diffusion training.
    
    Args:
        tokens: (B, seq_len) ground truth tokens
        mask_id: ID of the MASK token
        mask_ratio_min: minimum masking ratio
        mask_ratio_max: maximum masking ratio
    
    Returns:
        masked_tokens: tokens with some positions masked
        mask_positions: bool tensor indicating which positions are masked
    """
    B, N = tokens.shape
    device = tokens.device
    
    # Sample different mask ratio for each sample in batch
    mask_ratios = torch.rand(B, device=device) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
    
    # Create masks
    rand_values = torch.rand(B, N, device=device)
    mask_positions = rand_values < mask_ratios.unsqueeze(1)
    
    # Apply masks
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = mask_id
    
    return masked_tokens, mask_positions


def add_uniform_absorbing_mixture_noise(
    tokens,
    mask_id,
    mask_ratio_min=0.2,
    mask_ratio_max=0.9,
    mixture_prob=0.1,
    vocab_size=10,
):
    """
    Apply absorbing noise, then uniformly replace a subset of non-masked tokens.
    
    Non-masked tokens are replaced with random tokens (excluding the original token
    and the mask token) with probability `mixture_prob`.
    """
    masked_tokens, mask_positions = add_absorbing_noise(
        tokens, mask_id, mask_ratio_min=mask_ratio_min, mask_ratio_max=mask_ratio_max
    )
    
    if mixture_prob <= 0.0:
        mixture_positions = torch.zeros_like(mask_positions)
        return masked_tokens, mask_positions, mixture_positions
    
    device = tokens.device
    replace_candidates = masked_tokens != mask_id
    if not replace_candidates.any():
        mixture_positions = torch.zeros_like(mask_positions)
        return masked_tokens, mask_positions, mixture_positions
    
    rand_values = torch.rand_like(masked_tokens, dtype=torch.float32)
    replace_mask = replace_candidates & (rand_values < mixture_prob)
    
    if not replace_mask.any():
        mixture_positions = torch.zeros_like(mask_positions)
        return masked_tokens, mask_positions, mixture_positions
    
    original_tokens = masked_tokens[replace_mask]
    
    new_tokens = torch.randint(
        0, vocab_size - 1, size=original_tokens.shape, device=device, dtype=torch.long
    )
    new_tokens = new_tokens + (new_tokens >= mask_id).long()
    
    clashes = new_tokens == original_tokens
    while clashes.any():
        fresh = torch.randint(
            0,
            vocab_size - 1,
            size=(clashes.sum().item(),),
            device=device,
            dtype=torch.long,
        )
        fresh = fresh + (fresh >= mask_id).long()
        new_tokens[clashes] = fresh
        clashes = new_tokens == original_tokens
    
    masked_tokens[replace_mask] = new_tokens
    mixture_positions = torch.zeros_like(mask_positions)
    mixture_positions[replace_mask] = True
    
    return masked_tokens, mask_positions, mixture_positions


def compute_absorbing_loss(logits, targets, mask_positions):
    """
    Compute absorbing loss (cross-entropy on masked positions).
    
    Args:
        logits: (B, seq_len, vocab_size)
        targets: (B, seq_len)
        mask_positions: (B, seq_len) bool tensor
    
    Returns:
        loss: scalar loss
        accuracy: scalar accuracy on masked positions
    """
    # Flatten
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    mask_flat = mask_positions.reshape(-1)
    
    # Select only masked positions
    logits_masked = logits_flat[mask_flat]
    targets_masked = targets_flat[mask_flat]
    
    if len(targets_masked) == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    
    # Compute loss
    loss = F.cross_entropy(logits_masked, targets_masked)
    
    # Compute accuracy
    preds = logits_masked.argmax(dim=-1)
    accuracy = (preds == targets_masked).float().mean()
    
    return loss, accuracy


def evaluate(
    model,
    dataloader,
    device,
    mask_id,
    loss_mode='absorbing',
    mixture_prob=0.1,
    mixture_loss_weight=1.0,
    clean_loss_weight=1.0,
):
    """Evaluate model on validation set."""
    valid_modes = {
        'absorbing',
        'uniform_absorbing_mixture',
        'uniform_absorbing_mixture_with_clean',
    }
    if loss_mode not in valid_modes:
        raise NotImplementedError(f"Loss mode '{loss_mode}' not implemented.")
    model.eval()
    total_loss = 0.0
    total_acc_union = 0.0
    # For mixture diagnostics
    total_mask_loss = 0.0
    total_mask_acc = 0.0
    total_mix_loss = 0.0
    total_mix_acc = 0.0
    total_clean_loss = 0.0
    total_clean_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            solutions = batch['solution'].to(device)
            
            # Mask with fixed ratio
            if loss_mode == 'absorbing':
                masked_tokens, mask_positions = add_absorbing_noise(
                    solutions, mask_id=mask_id, mask_ratio_min=0.5, mask_ratio_max=0.5
                )
                # Forward pass
                output = model(masked_tokens)
                # Loss on masked positions
                loss, acc = compute_absorbing_loss(output.logits, solutions, mask_positions)
                total_loss += loss.item()
                total_acc_union += acc.item()
                num_batches += 1
            elif loss_mode in {
                'uniform_absorbing_mixture',
                'uniform_absorbing_mixture_with_clean',
            }:
                vocab_size = getattr(model.config, 'vocab_size', 10)
                masked_tokens, mask_positions, mixture_positions = add_uniform_absorbing_mixture_noise(
                    solutions,
                    mask_id=mask_id,
                    mask_ratio_min=0.5,
                    mask_ratio_max=0.5,
                    mixture_prob=mixture_prob,
                    vocab_size=vocab_size,
                )
                # Forward pass
                output = model(masked_tokens)
                # Per-component losses/acc
                mask_loss, mask_acc = compute_absorbing_loss(
                    output.logits, solutions, mask_positions
                )
                mix_loss, mix_acc = compute_absorbing_loss(
                    output.logits, solutions, mixture_positions
                )
                base_union = mask_positions | mixture_positions
                clean_positions = ~base_union
                clean_loss = torch.tensor(0.0, device=device)
                clean_acc = torch.tensor(0.0, device=device)
                if loss_mode == 'uniform_absorbing_mixture_with_clean':
                    clean_loss, clean_acc = compute_absorbing_loss(
                        output.logits, solutions, clean_positions
                    )
                if loss_mode == 'uniform_absorbing_mixture_with_clean':
                    acc_positions = torch.ones_like(base_union, dtype=torch.bool)
                else:
                    acc_positions = base_union
                _, acc_union = compute_absorbing_loss(
                    output.logits, solutions, acc_positions
                )
                combined = mask_loss + mixture_loss_weight * mix_loss
                if loss_mode == 'uniform_absorbing_mixture_with_clean':
                    combined = combined + clean_loss_weight * clean_loss
                total_loss += combined.item()
                total_acc_union += acc_union.item()
                total_mask_loss += mask_loss.item()
                total_mask_acc += mask_acc.item()
                total_mix_loss += mix_loss.item()
                total_mix_acc += mix_acc.item()
                if loss_mode == 'uniform_absorbing_mixture_with_clean':
                    total_clean_loss += clean_loss.item()
                    total_clean_acc += clean_acc.item()
                num_batches += 1
            else:
                raise NotImplementedError(f"Loss mode '{loss_mode}' not implemented.")
    
    model.train()
    if loss_mode == 'absorbing':
        return total_loss / num_batches, total_acc_union / num_batches, {}
    elif loss_mode == 'uniform_absorbing_mixture':
        extras = {
            'mask_loss': total_mask_loss / num_batches,
            'mask_acc': total_mask_acc / num_batches,
            'mix_loss': total_mix_loss / num_batches,
            'mix_acc': total_mix_acc / num_batches,
        }
        return total_loss / num_batches, total_acc_union / num_batches, extras
    elif loss_mode == 'uniform_absorbing_mixture_with_clean':
        extras = {
            'mask_loss': total_mask_loss / num_batches,
            'mask_acc': total_mask_acc / num_batches,
            'mix_loss': total_mix_loss / num_batches,
            'mix_acc': total_mix_acc / num_batches,
            'clean_loss': total_clean_loss / num_batches,
            'clean_acc': total_clean_acc / num_batches,
        }
        return total_loss / num_batches, total_acc_union / num_batches, extras
    else:
        raise NotImplementedError(f"Loss mode '{loss_mode}' not implemented.")


def train(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset = SudokuDataset(args.data_dir, split='train')
    test_dataset = SudokuDataset(args.data_dir, split='test')
    mask_id = train_dataset.mask_id
    
    # Setup wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': {
                    'vocab_size': 10,  # 0: MASK, 1-9: digits (no EOL)
                    'seq_length': 81,
                    'hidden_dim': 512,
                    'num_layers': 12,
                    'num_heads': 8,
                    'mlp_ratio': 4,
                    'dropout': 0.1,
                },
                'training': {
                    'train_batch_size': args.train_batch_size,
                    'eval_batch_size': args.eval_batch_size,
                    'num_steps': args.num_steps,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'mask_ratio_min': args.mask_ratio_min,
                    'mask_ratio_max': args.mask_ratio_max,
                    'loss_mode': args.loss_mode,
                    'uniform_mixture_prob': args.uniform_mixture_prob,
                    'uniform_mixture_loss_weight': args.uniform_mixture_loss_weight,
                    'uniform_clean_loss_weight': args.uniform_clean_loss_weight,
                },
                'data': {
                    'train_size': len(train_dataset),
                    'test_size': len(test_dataset),
                }
            }
        )
        use_wandb = True
    else:
        use_wandb = False
        print("Running without wandb logging")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_sudoku_dit(vocab_size=10, seq_length=81)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
        eta_min=args.lr * 0.1
    )
    
    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"Train batch size: {args.train_batch_size}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mask ratio: [{args.mask_ratio_min}, {args.mask_ratio_max}]")
    if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
        print(f"Uniform mixture prob: {args.uniform_mixture_prob}")
    
    # Initial evaluation before training starts
    print("\nInitial evaluation before training...")
    initial_loss, initial_acc, initial_extras = evaluate(
        model,
        test_loader,
        device,
        mask_id=mask_id,
        loss_mode=args.loss_mode,
        mixture_prob=args.uniform_mixture_prob,
        mixture_loss_weight=args.uniform_mixture_loss_weight,
        clean_loss_weight=getattr(args, 'uniform_clean_loss_weight', 1.0),
    )
    print(f"Initial Test Loss: {initial_loss:.4f}, Test Acc: {initial_acc:.3f}")
    if use_wandb:
        init_log = {
            'test/loss': initial_loss,
            'test/accuracy': initial_acc,
            'test/step': 0,
        }
        if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
            init_log['test/uniform_mixture_prob'] = args.uniform_mixture_prob
        if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
            init_log['test/mask_loss'] = initial_extras.get('mask_loss', 0.0)
            init_log['test/mask_acc'] = initial_extras.get('mask_acc', 0.0)
            init_log['test/mix_loss'] = initial_extras.get('mix_loss', 0.0)
            init_log['test/mix_acc'] = initial_extras.get('mix_acc', 0.0)
        if args.loss_mode == 'uniform_absorbing_mixture_with_clean':
            init_log['test/uniform_clean_loss_weight'] = args.uniform_clean_loss_weight
            init_log['test/clean_loss'] = initial_extras.get('clean_loss', 0.0)
            init_log['test/clean_acc'] = initial_extras.get('clean_acc', 0.0)
        wandb.log(init_log, step=0)

    model.train()
    step = 0
    epoch = 0
    best_test_acc = initial_acc
    
    train_iterator = iter(train_loader)
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while step < args.num_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            # New epoch
            epoch += 1
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        solutions = batch['solution'].to(device)
        
        # Apply corruption based on loss mode
        if args.loss_mode == 'absorbing':
            masked_tokens, mask_positions = add_absorbing_noise(
                solutions,
                mask_id=mask_id,
                mask_ratio_min=args.mask_ratio_min,
                mask_ratio_max=args.mask_ratio_max,
            )
        elif args.loss_mode in {
            'uniform_absorbing_mixture',
            'uniform_absorbing_mixture_with_clean',
        }:
            masked_tokens, mask_positions, mixture_positions = add_uniform_absorbing_mixture_noise(
                solutions,
                mask_id=mask_id,
                mask_ratio_min=args.mask_ratio_min,
                mask_ratio_max=args.mask_ratio_max,
                mixture_prob=args.uniform_mixture_prob,
                vocab_size=model.config.vocab_size,
            )
        else:
            raise NotImplementedError(f"Loss mode '{args.loss_mode}' not implemented.")
        
        # Forward pass
        output = model(masked_tokens)
        
        # Compute loss
        if args.loss_mode == 'absorbing':
            loss, acc = compute_absorbing_loss(output.logits, solutions, mask_positions)
            mask_loss = loss
            mask_acc = acc
            mix_loss = torch.tensor(0.0, device=device)
            mix_acc = torch.tensor(0.0, device=device)
        else:
            # Two components
            mask_loss, mask_acc = compute_absorbing_loss(
                output.logits, solutions, mask_positions
            )
            mix_loss, mix_acc = compute_absorbing_loss(
                output.logits, solutions, mixture_positions
            )
            # Combined
            loss = mask_loss + args.uniform_mixture_loss_weight * mix_loss
            base_union = mask_positions | mixture_positions
            clean_loss = torch.tensor(0.0, device=device)
            clean_acc = torch.tensor(0.0, device=device)
            if args.loss_mode == 'uniform_absorbing_mixture_with_clean':
                clean_positions = ~base_union
                clean_loss, clean_acc = compute_absorbing_loss(
                    output.logits, solutions, clean_positions
                )
                loss = loss + args.uniform_clean_loss_weight * clean_loss
                acc_positions = torch.ones_like(base_union, dtype=torch.bool)
            else:
                acc_positions = base_union
            _, acc = compute_absorbing_loss(output.logits, solutions, acc_positions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % args.log_interval == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/accuracy': acc.item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/epoch': epoch,
                'train/step': step,
            }
            if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
                log_dict['train/uniform_mixture_prob'] = args.uniform_mixture_prob
                log_dict['train/mask_loss'] = mask_loss.item()
                log_dict['train/mix_loss'] = mix_loss.item()
                log_dict['train/mask_acc'] = mask_acc.item()
                log_dict['train/mix_acc'] = mix_acc.item()
            if args.loss_mode == 'uniform_absorbing_mixture_with_clean':
                log_dict['train/uniform_clean_loss_weight'] = args.uniform_clean_loss_weight
                log_dict['train/clean_loss'] = clean_loss.item()
                log_dict['train/clean_acc'] = clean_acc.item()
            
            if use_wandb:
                wandb.log(log_dict, step=step)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.3f}',
                'epoch': epoch
            })
        
        # Evaluation
        if step % args.eval_interval == 0 and step > 0:
            test_loss, test_acc, test_extras = evaluate(
                model,
                test_loader,
                device,
                mask_id=mask_id,
                loss_mode=args.loss_mode,
                mixture_prob=args.uniform_mixture_prob,
                mixture_loss_weight=args.uniform_mixture_loss_weight,
                clean_loss_weight=args.uniform_clean_loss_weight,
            )
            
            eval_dict = {
                'test/loss': test_loss,
                'test/accuracy': test_acc,
                'test/step': step,
            }
            if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
                eval_dict['test/uniform_mixture_prob'] = args.uniform_mixture_prob
                eval_dict['test/mask_loss'] = test_extras.get('mask_loss', 0.0)
                eval_dict['test/mask_acc'] = test_extras.get('mask_acc', 0.0)
                eval_dict['test/mix_loss'] = test_extras.get('mix_loss', 0.0)
                eval_dict['test/mix_acc'] = test_extras.get('mix_acc', 0.0)
            if args.loss_mode == 'uniform_absorbing_mixture_with_clean':
                eval_dict['test/uniform_clean_loss_weight'] = args.uniform_clean_loss_weight
                eval_dict['test/clean_loss'] = test_extras.get('clean_loss', 0.0)
                eval_dict['test/clean_acc'] = test_extras.get('clean_acc', 0.0)
            
            if use_wandb:
                wandb.log(eval_dict, step=step)
            
            print(f"\n[Step {step}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.3f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc,
                    'config': model.config.__dict__,
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                
                if use_wandb:
                    wandb.log({'best_test_acc': best_test_acc}, step=step)
                
                print(f"✓ Saved best model (acc={best_test_acc:.3f})")
        
        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_acc': best_test_acc,
                'config': model.config.__dict__,
            }, output_dir / f'checkpoint_{step}.pt')
            print(f"✓ Saved checkpoint at step {step}")
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final evaluation
    print("\n" + "="*50)
    print("Training complete! Running final evaluation...")
    test_loss, test_acc, test_extras = evaluate(
        model,
        test_loader,
        device,
        mask_id=mask_id,
        loss_mode=args.loss_mode,
        mixture_prob=args.uniform_mixture_prob,
        mixture_loss_weight=args.uniform_mixture_loss_weight,
        clean_loss_weight=getattr(args, 'uniform_clean_loss_weight', 1.0),
    )
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.3f}")
    print(f"Best Test Accuracy: {best_test_acc:.3f}")
    
    # Save final model
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'test_acc': test_acc,
        'best_test_acc': best_test_acc,
        'config': model.config.__dict__,
    }, output_dir / 'final_model.pt')
    print(f"✓ Saved final model to {output_dir / 'final_model.pt'}")
    
    # Finish wandb
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Sudoku Diffusion Language Model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: <loss_mode>_checkpoints)')
    
    # Model (default: 28.6M params)
    # These are defined in model.py, but can be overridden
    
    # Training
    parser.add_argument('--train-batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--num-steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--mask-ratio-min', type=float, default=0.2, help='Minimum mask ratio')
    parser.add_argument('--mask-ratio-max', type=float, default=0.9, help='Maximum mask ratio')
    parser.add_argument(
        '--uniform-mixture-prob',
        type=float,
        default=0.1,
        help='Token replacement probability for uniform_absorbing_mixture loss mode',
    )
    parser.add_argument(
        '--uniform-mixture-loss-weight',
        type=float,
        default=1.0,
        help='Loss weight for random-token replacement component in uniform_absorbing_mixture mode',
    )
    parser.add_argument(
        '--uniform-clean-loss-weight',
        type=float,
        default=1.0,
        help='Loss weight for clean tokens in uniform_absorbing_mixture_with_clean mode',
    )
    parser.add_argument('--loss-mode', type=str, default='absorbing', help='Training loss mode (e.g., absorbing)')
    
    # System
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--save-interval', type=int, default=10000, help='Save checkpoint every N steps')
    
    # Wandb
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='sudoku-diffusion-lm', help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*50)
    print("Sudoku Diffusion Language Model - Training")
    print("="*50)
    print(f"Data directory: {args.data_dir}")
    output_path = Path(args.output_dir) if args.output_dir is not None else Path(f'./{args.loss_mode}_checkpoints')
    if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
        if not 0.0 <= args.uniform_mixture_prob <= 1.0:
            raise ValueError("--uniform-mixture-prob must be in [0, 1]")
        prob_dir = f'prob_{args.uniform_mixture_prob}'
        if output_path.name != prob_dir:
            output_path = output_path / prob_dir
    args.output_dir = str(output_path)
    print(f"Output directory: {args.output_dir}")
    print(f"Train batch size: {args.train_batch_size}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Training steps: {args.num_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss mode: {args.loss_mode}")
    if args.loss_mode in {'uniform_absorbing_mixture', 'uniform_absorbing_mixture_with_clean'}:
        print(f"Uniform mixture prob: {args.uniform_mixture_prob}")
        print(f"Uniform mixture loss weight: {args.uniform_mixture_loss_weight}")
    print("="*50)
    
    train(args)


if __name__ == '__main__':
    main()

