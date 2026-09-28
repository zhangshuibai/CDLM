#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare absorbing results across multiple checkpoints.
Generates line plots showing board accuracy vs sampling steps for different
mask_ratio combinations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Set Times New Roman font for publication-quality figures
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # Use STIX for math text
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 22
matplotlib.rcParams['legend.fontsize'] = 22
matplotlib.rcParams['figure.dpi'] = 300


def load_results_file(result_dir: Path) -> pd.DataFrame:
    """Load absorbing results from CSV or JSON."""
    csv_path = result_dir / "absorbing_results.csv"
    json_path = result_dir / "absorbing_results.json"
    
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
            f"No absorbing_results found in {result_dir}"
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
    mask_ratios: List[float] = None,
    steps: List[int] = None,
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
    
    # Collect all unique mask_ratios from data
    if mask_ratios is not None:
        all_mask_ratios = mask_ratios
    else:
        all_mask_ratios = set()
        for df in results_dict.values():
            if 'mask_ratio' in df.columns:
                mask_ratios_data = df['mask_ratio'].dropna().unique()
                all_mask_ratios.update(mask_ratios_data)
        all_mask_ratios = sorted(all_mask_ratios)
    
    if not all_mask_ratios:
        print("No mask_ratio combinations found")
        return
    
    # Define colors for each model (exactly same as radar chart)
    model_colors = {
        'MDLM': '#d62728',  # Red - exactly same as radar chart
    }
    # Use different shades of blue for different CDLM variants
    mixed_colors = {
        'CDLM': '#1f77b4',    # Blue - exactly same as radar chart
    }
    default_colors = ['#d62728', '#1f77b4', '#F18F01', '#A23B72', '#1E5F8A', '#4A9FD8']
    
    model_names = list(results_dict.keys())
    
    # Create a separate figure for each (algorithm, parameter) combination
    for algorithm, param_str, param_tuple in sorted(all_combos):
        params_dict = dict(param_tuple)
        
        n_plots = len(all_mask_ratios)
        # Use 2x2 layout for 4 plots, otherwise 3 columns
        if n_plots == 4:
            n_cols = 2
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Reduce width, increase height
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each mask_ratio
        for idx, mask_ratio in enumerate(all_mask_ratios):
            ax = axes[idx]
            
            # Plot each model
            for model_idx, model_name in enumerate(model_names):
                df = results_dict[model_name]
                
                # Filter data for this mask_ratio, algorithm, and parameters
                subset = df[
                    (df['mask_ratio'] == mask_ratio) &
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
                
                # Filter by specified steps if provided
                if steps is not None:
                    step_data = step_data[step_data['step'].isin(steps)]
                
                if len(step_data) > 0:
                    # Get display name for legend (remove prob value)
                    display_name = map_model_name(model_name)
                    
                    # Get color: check specific model colors first, then mixed colors, then default
                    if display_name in model_colors:
                        model_color = model_colors[display_name]
                    elif display_name == 'CDLM':
                        # Use same blue color for all CDLM variants (consistent with radar chart)
                        model_color = mixed_colors.get('CDLM', '#1f77b4')
                    else:
                        model_color = default_colors[model_idx % len(default_colors)]
                    
                    # Use different line styles for different Mixed Objective variants
                    # Check original model_name to determine line style
                    linestyle = '-'
                    if 'absorbing' in model_name:
                        linestyle = '-'
                    elif 'prob_0.1' in model_name:
                        linestyle = '-'  # Solid for prob=0.1
                    elif 'prob_0.2' in model_name:
                        linestyle = '--'  # Dashed for prob=0.2
                    elif 'prob_0.02' in model_name:
                        linestyle = '-.'  # Dash-dot for prob=0.02
                    
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
                ax.set_xlabel("Sampling Steps", fontweight="bold", fontfamily='serif')
            else:
                ax.set_xlabel("")  # Empty label for non-bottom rows
            
            # Only show y-axis label on left column
            if is_left_col:
                ax.set_ylabel("Board Accuracy", fontweight="bold", fontfamily='serif')
            else:
                ax.set_ylabel("")  # Empty label for non-left columns
                # Hide y-axis tick labels for non-left columns
                ax.set_yticklabels([])
            
            ax.set_title(
                f"Mask Ratio={mask_ratio}",
                fontweight="bold",
                fontfamily='serif',
                pad=10,
            )
            ax.set_ylim(0.0, 1.0)
            
            # Set x-axis: Use specified steps or all available steps
            if steps is not None:
                step_values = sorted([s for s in steps if s in subset['step'].values]) if not subset.empty else steps
            else:
                step_values = sorted(subset['step'].unique()) if not subset.empty else [1, 2, 3, 4]
            
            if len(step_values) > 0:
                ax.set_xlim(min(step_values) - 0.5, max(step_values) + 0.5)
                # Show all step values as integers
                ax.set_xticks(step_values)
                ax.set_xticklabels([int(s) for s in step_values])
            
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
            
            # Only show legend on bottom-right subplot (last subplot)
            if idx == n_plots - 1:
                ax.legend(loc="best", prop={'family': 'serif'}, framealpha=0.9)
            
            # Set tick labels font
            for label in ax.get_xticklabels():
                label.set_fontfamily('serif')
            for label in ax.get_yticklabels():
                label.set_fontfamily('serif')
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Format algorithm and parameter name for filename
        algorithm_safe = algorithm.replace(':', '_').replace('-', '_')
        if params_dict:
            param_part = '_'.join([f"{k}_{v}".replace('.', '').replace('-', 'neg') for k, v in sorted(params_dict.items())])
            filename = f"sudoku_absorbing_{algorithm_safe}_{param_part}_comparison.pdf"
        else:
            filename = f"sudoku_absorbing_{algorithm_safe}_comparison.pdf"
        
        # Reduce spacing between subplots, leave space for legend at bottom
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.05, 1, 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PDF for publication
        output_path_pdf = output_dir / filename
        fig.savefig(output_path_pdf, format='pdf', bbox_inches="tight", dpi=300)
        
        plt.close(fig)
        print(f"Saved: {output_path_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare absorbing results across multiple checkpoints"
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
        "--mask-ratios",
        type=float,
        nargs="+",
        default=None,
        help="List of mask_ratios to display (e.g., 0.3 0.4 0.5 0.6). "
             "If not provided, uses all mask_ratios found in data",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="List of steps to display (e.g., 1 2 3 4 5 6 7 8). "
             "If not provided, uses all steps found in data",
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
    
    # Parse mask_ratios from command line or use None (will be auto-detected)
    mask_ratios = None
    if args.mask_ratios:
        mask_ratios = sorted(args.mask_ratios)
    
    # Parse steps from command line or use None (will be auto-detected)
    steps = None
    if args.steps:
        steps = sorted(args.steps)
    
    # Create comparison plots in subdirectory
    base_output_dir = Path(args.output_dir).resolve()
    output_dir = base_output_dir / "absorbing_completion_compare"
    plot_line_comparison(results_dict, output_dir, mask_ratios, steps)
    
    print(f"\nComparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

