#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare uniform_noise_diffusion results across multiple checkpoints.
Generates line plots showing board accuracy vs sampling steps for different
noise_ratio × editable_ratio combinations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set Times New Roman font for publication-quality figures
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # Use STIX for math text
matplotlib.rcParams['axes.labelsize'] = 26  # Increased for paper
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 24  # Increased for paper
matplotlib.rcParams['ytick.labelsize'] = 24  # Increased for paper
matplotlib.rcParams['legend.fontsize'] = 24  # Increased for paper
matplotlib.rcParams['figure.dpi'] = 300


def load_results_file(result_dir: Path) -> pd.DataFrame:
    """Load uniform_noise_diffusion results from CSV or JSON."""
    csv_path = result_dir / "uniform_noise_diffusion_results.csv"
    json_path = result_dir / "uniform_noise_diffusion_results.json"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Ensure step column exists (might be named 'steps' in some files)
        if 'steps' in df.columns and 'step' not in df.columns:
            df['step'] = df['steps']
        return df
    elif json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Ensure step column exists
        if 'steps' in df.columns and 'step' not in df.columns:
            df['step'] = df['steps']
        return df
    else:
        raise FileNotFoundError(
            f"No uniform_noise_diffusion_results found in {result_dir}"
        )


def map_model_name(model_name: str) -> str:
    """Map model names to display names."""
    if model_name == 'absorbing':
        return 'MDLM'
    elif model_name.startswith('prob_'):
        # Remove prob value, just show "CDLM"
        return 'CDLM'
    return model_name


def extract_model_name(result_dir: Path) -> str:
    """Extract a readable model name from the result directory path."""
    parts = result_dir.parts
    raw_name = None
    if "prob_" in str(result_dir):
        for part in parts:
            if part.startswith("prob_"):
                raw_name = part
                break
        if raw_name is None:
            raw_name = parts[-2] if len(parts) >= 2 else parts[-1]
    elif "checkpoints_absorbing" in str(result_dir):
        raw_name = "absorbing"
    else:
        raw_name = parts[-2] if len(parts) >= 2 else parts[-1]
    
    # Return raw name (will be mapped later for display)
    return raw_name


def get_algorithm_parameter_combinations(df: pd.DataFrame) -> List[Tuple[str, Dict]]:
    """Get all (algorithm, parameter) combinations from dataframe."""
    combinations = []
    
    for algorithm in df['algorithm'].unique():
        algo_subset = df[df['algorithm'] == algorithm]
        
        # Check for confidence_threshold parameter
        has_confidence = 'confidence_threshold' in algo_subset.columns
        has_mad_k = 'mad_k' in algo_subset.columns
        
        # Check which parameters actually have values (not all NaN)
        thresholds = []
        if has_confidence:
            thresholds = algo_subset['confidence_threshold'].dropna().unique()
        
        mad_ks = []
        if has_mad_k:
            mad_ks = algo_subset['mad_k'].dropna().unique()
        
        # Generate combinations based on which parameters have values
        if len(thresholds) > 0:
            # Has confidence_threshold values
            for threshold in sorted(thresholds):
                combinations.append((algorithm, {'confidence_threshold': threshold}))
        elif len(mad_ks) > 0:
            # Has mad_k values
            for mad_k in sorted(mad_ks):
                combinations.append((algorithm, {'mad_k': mad_k}))
        else:
            # No parameters with values
            combinations.append((algorithm, {}))
    
    return combinations


def plot_line_comparison(
    results_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    combinations: List[Tuple[float, float]] = None,
) -> None:
    """Plot line comparison charts for each algorithm and parameter combination separately."""
    # Collect all unique (algorithm, parameter) combinations from all models
    all_combos = set()
    for df in results_dict.values():
        if 'algorithm' in df.columns:
            combos = get_algorithm_parameter_combinations(df)
            for algo, params in combos:
                # Create a hashable representation
                param_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
                all_combos.add((algo, param_str, tuple(sorted(params.items()))))
    
    if not all_combos:
        print("No algorithm combinations found")
        return
    
    # Use provided combinations or collect all unique combinations from data
    if combinations is not None:
        all_combinations = combinations
    else:
        # Collect all unique (noise_ratio, editable_ratio) combinations
        all_combinations = set()
        for df in results_dict.values():
            if 'noise_ratio' in df.columns and 'editable_ratio' in df.columns:
                data_combinations = df[['noise_ratio', 'editable_ratio']].drop_duplicates()
                for _, row in data_combinations.iterrows():
                    all_combinations.add((row['noise_ratio'], row['editable_ratio']))
        all_combinations = sorted(all_combinations)
    
    if not all_combinations:
        print("No (noise_ratio, editable_ratio) combinations found")
        return
    
    # Sort combinations for consistent ordering (if not already sorted)
    if not isinstance(all_combinations, list):
        all_combinations = sorted(all_combinations)
    
    # Define colors for each model (exactly same as radar chart)
    model_colors = {
        'MDLM': '#d62728',  # Red - exactly same as radar chart
    }
    # Use different shades of blue for different CDLM variants
    # Note: After mapping, all CDLM variants will be "CDLM"
    mixed_colors = {
        'CDLM': '#1f77b4',    # Blue - exactly same as radar chart
    }
    default_colors = ['#d62728', '#1f77b4', '#F18F01', '#A23B72', '#1E5F8A', '#4A9FD8']
    
    model_names = list(results_dict.keys())
    
    # Create a separate figure for each (algorithm, parameter) combination
    for algorithm, param_str, param_tuple in sorted(all_combos):
        params_dict = dict(param_tuple)
        
        n_plots = len(all_combinations)
        n_cols = 3  # 3 columns for better layout
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each combination
        for idx, (noise_ratio, editable_ratio) in enumerate(all_combinations):
            ax = axes[idx]
            
            # Plot each model
            for model_idx, model_name in enumerate(model_names):
                df = results_dict[model_name]
                
                # Filter data for this combination, algorithm, and parameters
                subset = df[
                    (df['noise_ratio'] == noise_ratio) &
                    (df['editable_ratio'] == editable_ratio) &
                    (df['algorithm'] == algorithm)
                ].copy()
                
                # Apply parameter filters
                if 'confidence_threshold' in params_dict:
                    if 'confidence_threshold' in subset.columns:
                        subset = subset[subset['confidence_threshold'] == params_dict['confidence_threshold']]
                elif 'confidence_threshold' in subset.columns:
                    # For algorithms without confidence_threshold parameter, filter out rows that have it
                    subset = subset[subset['confidence_threshold'].isna()]
                
                if 'mad_k' in params_dict:
                    if 'mad_k' in subset.columns:
                        subset = subset[subset['mad_k'] == params_dict['mad_k']]
                elif 'mad_k' in subset.columns:
                    # For algorithms without mad_k parameter, filter out rows that have it
                    subset = subset[subset['mad_k'].isna()]
                
                if subset.empty:
                    continue
                
                # Group by step and get board_accuracy
                step_data = subset.groupby('step')['board_accuracy'].mean().reset_index()
                step_data = step_data.sort_values('step')
                
                if len(step_data) > 0:
                    # Get display name for legend (remove prob value)
                    display_name = map_model_name(model_name)
                    
                    # Get color: check specific model colors first, then mixed colors, then default
                    if display_name in model_colors:
                        model_color = model_colors[display_name]
                    elif display_name in mixed_colors:
                        model_color = mixed_colors[display_name]
                    else:
                        model_color = default_colors[model_idx % len(default_colors)]
                    
                    # Use different line styles for different Mixed Objective variants
                    # Check original model_name to determine line style
                    linestyle = '-'
                    if 'absorbing' in model_name:
                        linestyle = '-'
                    elif 'prob_0.2' in model_name:
                        linestyle = '--'
                    elif 'prob_0.02' in model_name:
                        linestyle = '-.'
                    
                    ax.plot(
                        step_data['step'],
                        step_data['board_accuracy'],
                        marker='o',
                        label=display_name,
                        color=model_color,
                        linewidth=2.5,
                        markersize=7,
                        alpha=0.8,
                        linestyle=linestyle,
                    )
            
            # Customize subplot
            # Determine position in grid
            is_bottom_row = idx >= (n_rows - 1) * n_cols
            is_left_col = idx % n_cols == 0
            is_right_col = idx % n_cols == n_cols - 1
            
            # Only show x-axis label on bottom row
            if is_bottom_row:
                ax.set_xlabel("Refined Steps", fontsize=26, fontweight="bold", fontfamily='serif')  # Changed to "Refined Steps" and increased font size
            else:
                ax.set_xlabel("")  # Empty label for non-bottom rows
            
            # Only show y-axis label on left column
            if is_left_col:
                ax.set_ylabel("Board Accuracy", fontsize=26, fontweight="bold", fontfamily='serif')  # Increased font size
            else:
                ax.set_ylabel("")  # Empty label for non-left columns
                # Hide y-axis tick labels for non-left columns
                ax.set_yticklabels([])
            
            ax.set_title(
                f"Noise={noise_ratio}, Editable={editable_ratio}",
                fontweight="bold",
                fontfamily='serif',
                pad=10,
            )
            ax.set_ylim(0.0, 0.9)
            
            # Set x-axis: actual data is 2,3,4 but display as 1,2,3
            # Collect all step values from all models to determine x-axis range
            all_steps = set()
            for model_name in model_names:
                df_model = results_dict[model_name]
                subset_model = df_model[
                    (df_model['noise_ratio'] == noise_ratio) &
                    (df_model['editable_ratio'] == editable_ratio) &
                    (df_model['algorithm'] == algorithm)
                ].copy()
                # Apply parameter filters
                if 'confidence_threshold' in params_dict:
                    if 'confidence_threshold' in subset_model.columns:
                        subset_model = subset_model[subset_model['confidence_threshold'] == params_dict['confidence_threshold']]
                elif 'confidence_threshold' in subset_model.columns:
                    subset_model = subset_model[subset_model['confidence_threshold'].isna()]
                if 'mad_k' in params_dict:
                    if 'mad_k' in subset_model.columns:
                        subset_model = subset_model[subset_model['mad_k'] == params_dict['mad_k']]
                elif 'mad_k' in subset_model.columns:
                    subset_model = subset_model[subset_model['mad_k'].isna()]
                if not subset_model.empty and 'step' in subset_model.columns:
                    all_steps.update(subset_model['step'].dropna().unique())
            
            if all_steps:
                min_step = min(all_steps)
                max_step = max(all_steps)
                # Set xlim to start from first data point, not before it
                ax.set_xlim(min_step - 0.3, max_step + 0.3)  # Small padding around data points
                ax.set_xticks(sorted(all_steps))  # Actual tick positions (data values)
                # Display labels shifted by -1 (2->1, 3->2, 4->3)
                ax.set_xticklabels([int(s) - 1 for s in sorted(all_steps)])
            else:
                # Fallback: use default range if no data
                ax.set_xlim(1.5, 4.5)
                ax.set_xticks([2, 3, 4])
                ax.set_xticklabels([1, 2, 3])
            
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
            
            # Only show legend on bottom-right subplot (last subplot)
            if idx == n_plots - 1:
                ax.legend(loc="best", prop={'family': 'serif'}, framealpha=0.9, fontsize=24)  # Increased for paper
            
            # Set tick labels font (increased size for paper)
            for label in ax.get_xticklabels():
                label.set_fontfamily('serif')
                label.set_fontsize(24)  # Increased for paper
            for label in ax.get_yticklabels():
                label.set_fontfamily('serif')
                label.set_fontsize(24)  # Increased for paper
            ax.tick_params(axis='x', labelsize=24)  # Increased for paper
            ax.tick_params(axis='y', labelsize=24)  # Increased for paper
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Format algorithm and parameter name for filename
        algorithm_safe = algorithm.replace(':', '_').replace('-', '_')
        if params_dict:
            param_part = '_'.join([f"{k}_{v}".replace('.', '').replace('-', 'neg') for k, v in sorted(params_dict.items())])
            filename = f"sudoku_uniform_noise_diffusion_{algorithm_safe}_{param_part}_comparison.pdf"
        else:
            filename = f"sudoku_uniform_noise_diffusion_{algorithm_safe}_comparison.pdf"
        
        # Reduce spacing between subplots
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PDF for publication
        output_path_pdf = output_dir / filename
        fig.savefig(output_path_pdf, format='pdf', bbox_inches="tight", dpi=300)
        
        plt.close(fig)
        
        # Print parameter info
        param_info = ', '.join([f"{k}={v}" for k, v in sorted(params_dict.items())]) if params_dict else "no parameters"
        print(f"Line comparison plots for {algorithm} ({param_info}) saved to: {output_path_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uniform_noise_diffusion results across multiple checkpoints"
    )
    parser.add_argument(
        "--result-dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of result directory paths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compare_results_all",
        help="Output directory for comparison plots",
    )
    parser.add_argument(
        "--combinations",
        type=str,
        nargs="+",
        default=None,
        help="List of noise_ratio,editable_ratio combinations (e.g., '0.1,0.4' '0.1,0.5' '0.2,0.4'). "
             "If not provided, uses default: (0.1,0.4) (0.1,0.5) (0.1,0.6) (0.2,0.4) (0.2,0.5) (0.2,0.6)",
    )
    args = parser.parse_args()
    
    # Load results from all directories
    results_dict = {}
    for result_dir_str in args.result_dirs:
        result_dir = Path(result_dir_str).resolve()
        if not result_dir.exists():
            print(f"Warning: Result directory not found: {result_dir}")
            continue
        
        try:
            df = load_results_file(result_dir)
            model_name = extract_model_name(result_dir)
            results_dict[model_name] = df
            print(f"Loaded results from {result_dir} (model: {model_name})")
        except Exception as e:
            print(f"Error loading {result_dir}: {e}")
            continue
    
    if not results_dict:
        print("No valid results found")
        return
    
    # Parse combinations from command line or use default
    if args.combinations:
        combinations = []
        for combo_str in args.combinations:
            try:
                noise_str, editable_str = combo_str.split(',')
                combinations.append((float(noise_str), float(editable_str)))
            except ValueError:
                print(f"Warning: Invalid combination format '{combo_str}'. Expected format: 'noise,editable' (e.g., '0.1,0.4')")
                continue
        if not combinations:
            print("No valid combinations provided, using default")
            combinations = None
    else:
        # Default combinations
        combinations = [
            (0.1, 0.4), (0.1, 0.5), (0.1, 0.6),
            (0.2, 0.4), (0.2, 0.5), (0.2, 0.6),
        ]
    
    # Create comparison plots in subdirectory
    base_output_dir = Path(args.output_dir).resolve()
    output_dir = base_output_dir / "uniform_noise_diffusion_compare"
    plot_line_comparison(results_dict, output_dir, combinations)
    
    print(f"\nComparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

