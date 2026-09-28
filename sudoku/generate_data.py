#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Sudoku training/test data for Diffusion Language Model
Generates 48k training + 2k test puzzles and solutions as text files
"""

import numpy as np
import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from advanced_sudoku_generator import AdvancedSudokuGenerator


def generate_puzzle_task(task_id):
    """Generate a single puzzle with solution."""
    difficulty = task_id % 3  # 0=easy, 1=medium, 2=hard
    difficulty_map = {0: 'easy', 1: 'medium', 2: 'hard'}
    clue_map = {0: 40, 1: 35, 2: 30}
    
    generator = AdvancedSudokuGenerator()
    puzzle, solution = generator.generate_professional_sudoku(
        min_clues=clue_map[difficulty],
        symmetry=False,
        required_difficulty=difficulty_map[difficulty]
    )
    
    return puzzle, solution


def grid_to_sequence(grid):
    """
    Convert 9x9 grid to sequence of 81 tokens.
    No EOL tokens - position embedding encodes row/col structure.
    Format: [r1c1, r1c2, ..., r1c9, r2c1, ..., r9c9]
    """
    return grid.flatten().tolist()


def save_dataset(puzzles, solutions, output_dir, split_name):
    """Save dataset in various formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to sequences
    puzzle_seqs = [grid_to_sequence(p) for p in puzzles]
    solution_seqs = [grid_to_sequence(s) for s in solutions]
    
    # Save as numpy arrays
    np.save(output_dir / f'{split_name}_puzzles.npy', np.array(puzzle_seqs, dtype=np.int16))
    np.save(output_dir / f'{split_name}_solutions.npy', np.array(solution_seqs, dtype=np.int16))
    
    # Save as text (one example per line, space-separated tokens)
    with open(output_dir / f'{split_name}_puzzles.txt', 'w') as f:
        for seq in puzzle_seqs:
            f.write(' '.join(map(str, seq)) + '\n')
    
    with open(output_dir / f'{split_name}_solutions.txt', 'w') as f:
        for seq in solution_seqs:
            f.write(' '.join(map(str, seq)) + '\n')
    
    # Save metadata
    metadata = {
        'num_samples': len(puzzles),
        'seq_length': 81,
        'vocab_size': 10,  # 0: MASK/empty, 1-9: digits (no EOL)
        'format': '81 cells (0=MASK/empty, 1-9=digits), position embedding for structure',
        'split': split_name
    }
    
    with open(output_dir / f'{split_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {len(puzzles)} {split_name} samples to {output_dir}")


def dataset_exists(output_dir, split_name, expected_size, required_seq_length=81, required_vocab_size=10):
    """Check if dataset already exists with at least the requested size."""
    output_dir = Path(output_dir)
    puzzles_path = output_dir / f'{split_name}_puzzles.npy'
    solutions_path = output_dir / f'{split_name}_solutions.npy'
    metadata_path = output_dir / f'{split_name}_metadata.json'

    if not (puzzles_path.exists() and solutions_path.exists() and metadata_path.exists()):
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    num_samples = metadata.get('num_samples')
    seq_length = metadata.get('seq_length')
    vocab_size = metadata.get('vocab_size')

    if num_samples is None or seq_length is None or vocab_size is None:
        return False

    if num_samples < expected_size:
        return False

    if seq_length != required_seq_length or vocab_size != required_vocab_size:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate Sudoku dataset for Diffusion LM')
    parser.add_argument('--train-size', type=int, default=48000, help='Number of training samples')
    parser.add_argument('--test-size', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--force', action='store_true', help='Regenerate data even if existing files are found')
    args = parser.parse_args()
    
    num_workers = args.num_workers or cpu_count()
    print(f"Generating Sudoku dataset with {num_workers} workers...")
    print(f"Train: {args.train_size}, Test: {args.test_size}")
    
    # Generate training data
    if dataset_exists(args.output_dir, 'train', args.train_size) and not args.force:
        print("\n[1/2] Training data already exists with sufficient size. Skipping generation (use --force to regenerate).")
    else:
        print("\n[1/2] Generating training data...")
        train_tasks = list(range(args.train_size))
        with Pool(processes=num_workers) as pool:
            train_results = list(tqdm(
                pool.imap(generate_puzzle_task, train_tasks),
                total=args.train_size,
                desc="Training"
            ))

        train_puzzles, train_solutions = zip(*train_results)
        save_dataset(train_puzzles, train_solutions, args.output_dir, 'train')
    
    # Generate test data
    if dataset_exists(args.output_dir, 'test', args.test_size) and not args.force:
        print("\n[2/2] Test data already exists with sufficient size. Skipping generation (use --force to regenerate).")
    else:
        print("\n[2/2] Generating test data...")
        test_tasks = list(range(args.train_size, args.train_size + args.test_size))
        with Pool(processes=num_workers) as pool:
            test_results = list(tqdm(
                pool.imap(generate_puzzle_task, test_tasks),
                total=args.test_size,
                desc="Testing"
            ))

        test_puzzles, test_solutions = zip(*test_results)
        save_dataset(test_puzzles, test_solutions, args.output_dir, 'test')
    
    print(f"\n✅ Dataset generation complete!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Vocabulary: 0 (MASK/empty), 1-9 (digits)")
    print(f"   Sequence length: 81 tokens (9×9 grid, no EOL)")
    print(f"   Note: Position embedding encodes row/col structure")


if __name__ == '__main__':
    main()

