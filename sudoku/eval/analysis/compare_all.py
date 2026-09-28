#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare uniform_noise_only results across multiple checkpoints.
Generates a grouped bar chart comparing clean vs noise token confidence
for each model at different noise ratios.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

# Set Times New Roman font for publication-quality figures
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # Use STIX for math text
matplotlib.rcParams['axes.labelsize'] = 22  # Increased for paper
matplotlib.rcParams['axes.titlesize'] = 17
matplotlib.rcParams['xtick.labelsize'] = 18  # Increased for paper
matplotlib.rcParams['ytick.labelsize'] = 18  # Increased for paper
matplotlib.rcParams['legend.fontsize'] = 17  # Increased for paper
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.dpi'] = 300


def load_results_file(result_dir: Path) -> pd.DataFrame:
    """Load uniform_noise_only results from CSV or JSON."""
    csv_path = result_dir / "uniform_noise_only_results.csv"
    json_path = result_dir / "uniform_noise_only_results.json"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise FileNotFoundError(
            f"No uniform_noise_only_results found in {result_dir}"
        )


def map_model_name(model_name: str) -> str:
    """Map model names to display names."""
    mapping = {
        'absorbing': 'MDLM',
        'prob_0.1': 'CDLM',
        'prob_0.2': 'CDLM',
        'prob_0.02': 'CDLM',
    }
    # Check for exact match first
    if model_name in mapping:
        return mapping[model_name]
    # Check for partial match (e.g., prob_0.1 matches prob_0.1)
    for key, value in mapping.items():
        if key in model_name or model_name in key:
            return value
    return model_name


def extract_model_name(result_dir: Path) -> str:
    """Extract a readable model name from the result directory path."""
    # Try to extract meaningful parts from the path
    parts = result_dir.parts
    raw_name = None
    if "prob_" in str(result_dir):
        # Extract prob value and checkpoint type
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
    
    # Apply mapping
    return map_model_name(raw_name)


def format_value(value: float, threshold: float = 0.001) -> str:
    """Format value with scientific notation for small values."""
    if value == 0.0:
        return "0"
    elif abs(value) < threshold:
        # Use scientific notation for small values
        # Format as scientific notation string first
        sci_str = f"{value:.2e}"
        parts = sci_str.split('e')
        coeff = float(parts[0])
        exp = int(parts[1])
        # Ensure coefficient is between 1 and 10
        if abs(coeff) >= 10:
            coeff /= 10
            exp += 1
        elif abs(coeff) < 1 and coeff != 0:
            coeff *= 10
            exp -= 1
        return f"{coeff:.2f}$\\times$10$^{{{exp}}}$"
    else:
        return f"{value:.3f}"


def plot_comparison(
    results_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot comparison chart for all models."""
    # Collect all noise ratios
    all_noise_ratios = set()
    for df in results_dict.values():
        if "noise_ratio" in df.columns:
            all_noise_ratios.update(df["noise_ratio"].unique())
    all_noise_ratios = sorted(all_noise_ratios)
    
    if not all_noise_ratios:
        print("No noise_ratio data found in any results")
        return
    
    # Prepare data for plotting
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_ratios = len(all_noise_ratios)
    
    # Bar width and spacing configuration
    bar_width = 0.35  # Width of each bar (clean or noise)
    gap_between_pairs = 0.15  # Gap between clean+noise pair and next model's pair
    gap_between_ratios = 0.5  # Gap between different noise_ratio groups
    
    # Calculate x positions for each group
    x_positions = {}  # (noise_ratio, model_name) -> (clean_x, noise_x)
    x_ticks = []
    x_tick_labels = []
    
    current_x = 0
    for noise_ratio in all_noise_ratios:
        group_start = current_x
        for model_idx, model_name in enumerate(model_names):
            clean_x = current_x
            noise_x = current_x + bar_width
            x_positions[(noise_ratio, model_name)] = (clean_x, noise_x)
            current_x += bar_width * 2  # Two bars (clean + noise)
            if model_idx < n_models - 1:
                current_x += gap_between_pairs  # Gap before next model
        # Center of this noise_ratio group
        group_center = (group_start + current_x - gap_between_pairs) / 2
        x_ticks.append(group_center)
        x_tick_labels.append(f"{noise_ratio}")
        current_x += gap_between_ratios
    
    # Create the plot with dual y-axes for better visualization of small values
    fig, ax1 = plt.subplots(figsize=(max(12, n_models * n_ratios * 1.2), 7))
    ax2 = ax1.twinx()  # Create second y-axis for log scale if needed
    
    # Collect data
    data = {}  # (noise_ratio, model_name) -> (clean_val, noise_val, ratio)
    
    for noise_ratio in all_noise_ratios:
        for model_name in model_names:
            df = results_dict[model_name]
            subset = df[df["noise_ratio"] == noise_ratio]
            
            if not subset.empty:
                clean_conf = float(subset["clean_token_confidence"].mean())
                noise_conf = float(subset["noise_token_confidence"].mean())
                ratio = clean_conf / noise_conf if noise_conf > 1e-8 else float('inf')
            else:
                clean_conf = 0.0
                noise_conf = 0.0
                ratio = 0.0
            
            data[(noise_ratio, model_name)] = (clean_conf, noise_conf, ratio)
    
    # Define colors for each model (exactly same as radar chart in plot_merged_radar_bar.py)
    # MDLM: red, CDLM: blue - must match exactly
    model_colors = {
        'MDLM': '#d62728',  # Red - exactly same as radar chart
        'CDLM': '#1f77b4',  # Blue - exactly same as radar chart
        'Absorbing Objective': '#d62728',  # Legacy support
        'Mixture Objective': '#1f77b4',     # Legacy support
        'Mixed Objective': '#1f77b4',      # Legacy support
    }
    # Fallback colors if model name not in mapping
    default_colors = ['#d62728', '#1f77b4', '#F18F01', '#A23B72']
    
    # Colors for clean and noise (lighter/darker variants)
    # Use full opacity for noise bars to match radar chart colors exactly
    clean_alpha = 0.9
    noise_alpha = 1.0  # Full opacity to match radar chart colors exactly
    
    # Plot bars for each model separately to enable different colors
    for model_idx, model_name in enumerate(model_names):
        model_color = model_colors.get(model_name, default_colors[model_idx % len(default_colors)])
        
        clean_x_list = []
        clean_y_list = []
        noise_x_list = []
        noise_y_list = []
        
        for noise_ratio in all_noise_ratios:
            clean_x, noise_x = x_positions[(noise_ratio, model_name)]
            clean_val, noise_val, ratio = data[(noise_ratio, model_name)]
            
            clean_x_list.append(clean_x)
            clean_y_list.append(clean_val)
            noise_x_list.append(noise_x)
            noise_y_list.append(noise_val)
        
        # Plot clean bars with lighter color
        clean_color_rgb = mcolors.to_rgb(model_color)
        clean_color_light = mcolors.to_hex((clean_color_rgb[0] * 0.7 + 0.3, 
                                           clean_color_rgb[1] * 0.7 + 0.3, 
                                           clean_color_rgb[2] * 0.7 + 0.3))
        
        # Plot bars with model-specific colors
        # Use lighter edge color to avoid making bars appear too dark
        edge_color = 'gray'  # Lighter edge color instead of black
        ax1.bar(clean_x_list, clean_y_list, width=bar_width, 
               label=f"{model_name} (Clean)" if model_idx == 0 else "", 
               color=clean_color_light, alpha=clean_alpha, 
               edgecolor=edge_color, linewidth=0.5)
        ax1.bar(noise_x_list, noise_y_list, width=bar_width, 
               label=f"{model_name} (Noise)" if model_idx == 0 else "", 
               color=model_color, alpha=noise_alpha, 
               edgecolor=edge_color, linewidth=0.5, hatch='///')
    
    # Annotate bars with values and ratios
    for noise_ratio in all_noise_ratios:
        for model_name in model_names:
            clean_x, noise_x = x_positions[(noise_ratio, model_name)]
            clean_val, noise_val, ratio = data[(noise_ratio, model_name)]
            
            # Annotate clean bar value
            if clean_val > 0:
                clean_text = format_value(clean_val)
                # For log scale, use multiplicative offset
                offset = clean_val * 1.1 if clean_val > 0.01 else clean_val * 2
                ax1.text(
                    clean_x,
                    offset,
                    clean_text,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontfamily='serif',
                )
            
            # Annotate noise bar value (closer to bar for better aesthetics)
            if noise_val > 0:
                noise_text = format_value(noise_val)
                # For log scale, use smaller multiplicative offset to bring text closer
                offset = noise_val * 1.2 if noise_val > 0.01 else noise_val * 1.2
                # Move more to the right to avoid overlap with adjacent values
                # For small values (like 1.57e-4, 2.69e-4), move further right
                if noise_val < 0.001:
                    noise_x_offset = noise_x + bar_width * 0.3  # Move more right for small values
                else:
                    noise_x_offset = noise_x + bar_width * 0.1  # Normal offset for larger values
                ax1.text(
                    noise_x_offset,
                    offset,
                    noise_text,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontfamily='serif',
                )
            
            # Annotate ratio above the pair (moved up to avoid overlap with bar value)
            if ratio > 0 and ratio != float('inf'):
                ratio_text = f"{ratio:.1f}x"
                # Position slightly to the right, aligned with noise bar center
                ratio_x_offset = noise_x + bar_width * 0.2
                # Position above noise value with increased spacing to avoid overlap
                noise_val_offset = noise_val * 1.2 if noise_val > 0.01 else noise_val * 1.2
                offset = noise_val_offset * 1.8  # Increased from 1.6 to 1.8 to move up more
                ax1.text(
                    ratio_x_offset,
                    offset,
                    ratio_text,
                    ha="center",
                    va="bottom",
                    fontsize=15,  # Increased from 13 to 15 for better visibility
                    fontweight="bold",
                    fontfamily='serif',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow", alpha=0.4, edgecolor='black', linewidth=0.5),
                )
            elif ratio == float('inf'):
                # Position slightly to the right and above noise value
                ratio_x_offset = noise_x + bar_width * 0.2
                noise_val_offset = noise_val * 1.2 if noise_val > 0.01 else noise_val * 1.2
                offset = noise_val_offset * 1.8  # Increased from 1.6 to 1.8 to move up more
                ax1.text(
                    ratio_x_offset,
                    offset,
                    "∞",
                    ha="center",
                    va="bottom",
                    fontsize=15,  # Increased from 13 to 15 for better visibility
                    fontweight="bold",
                    fontfamily='serif',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow", alpha=0.4, edgecolor='black', linewidth=0.5),
                )
    
    # Set x-axis
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, fontfamily='serif', fontsize=18)  # Increased for paper
    # Normal padding for x-axis (no model names below)
    ax1.tick_params(axis='x', pad=8, labelsize=18)  # Increased for paper
    ax1.set_xlabel("Noise Ratio", fontsize=22, fontweight="bold", fontfamily='serif', labelpad=8)  # Increased for paper
    ax1.set_ylabel("Token Confidence (log scale)", fontsize=22, fontweight="bold", fontfamily='serif')  # Increased for paper
    # Title removed as requested
    
    # Remove model name labels - they will be shown in legend instead
    # Calculate max value for y-axis scaling
    all_values = []
    for noise_ratio in all_noise_ratios:
        for model_name in model_names:
            clean_val, noise_val, _ = data[(noise_ratio, model_name)]
            all_values.extend([clean_val, noise_val])
    max_val = max(all_values) if all_values else 1.0
    
    # Add vertical lines to separate noise_ratio groups
    current_x = 0
    for noise_ratio_idx, noise_ratio in enumerate(all_noise_ratios):
        group_start = current_x
        for model_idx, model_name in enumerate(model_names):
            clean_x, noise_x = x_positions[(noise_ratio, model_name)]
            current_x = noise_x + bar_width
            if model_idx < n_models - 1:
                current_x += gap_between_pairs
        # Draw separator line after the group (except for the last one)
        if noise_ratio_idx < n_ratios - 1:
            ax1.axvline(x=current_x + gap_between_ratios / 2, color='gray', 
                      linestyle='--', linewidth=1, alpha=0.5)
        current_x += gap_between_ratios
    
    # Format y-axis for log scale
    from matplotlib.ticker import LogFormatterSciNotation
    ax1.yaxis.set_major_formatter(LogFormatterSciNotation())
    
    # Create custom legend entries for all models
    from matplotlib.patches import Rectangle
    legend_elements = []
    for model_idx, model_name in enumerate(model_names):
        model_color = model_colors.get(model_name, default_colors[model_idx % len(default_colors)])
        clean_color_rgb = mcolors.to_rgb(model_color)
        clean_color_light = mcolors.to_hex((clean_color_rgb[0] * 0.7 + 0.3, 
                                           clean_color_rgb[1] * 0.7 + 0.3, 
                                           clean_color_rgb[2] * 0.7 + 0.3))
        
        # Clean bar - use gray edge to match bars
        legend_elements.append(Rectangle((0, 0), 1, 1, 
                                         facecolor=clean_color_light, 
                                         alpha=clean_alpha,
                                         edgecolor='gray', linewidth=0.5,
                                         label=f"{model_name} (Clean)"))
        # Noise bar with hatch - use gray edge to match bars
        noise_rect = Rectangle((0, 0), 1, 1, 
                              facecolor=model_color, 
                              alpha=noise_alpha,
                              edgecolor='gray', linewidth=0.5,
                              hatch='///',
                              label=f"{model_name} (Noise)")
        legend_elements.append(noise_rect)
    
    # Place legend exactly at upper right corner, moved up to avoid blocking annotations
    # Use bbox_to_anchor slightly above 1.0 to move legend up
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=17,  # Increased for paper
              prop={'family': 'serif'}, ncol=1, framealpha=0.95,
              edgecolor='black', fancybox=True, shadow=False,
              bbox_to_anchor=(1.0, 1.02))  # Moved up (1.02) to avoid blocking annotations
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0, which='both')
    
    # Use log scale for Y-axis to better visualize small noise values
    # This makes both large and small values visible
    ax1.set_yscale('log')
    # Increase top limit significantly to avoid overlap with highest bar value (0.539) and accommodate legend
    ax1.set_ylim(bottom=1e-5, top=max_val * 3.5 if max_val > 0 else 1.0)  # Increased to 3.5 to provide more space for legend
    
    # Hide the second axis (we only use it if needed for dual scale)
    ax2.set_visible(False)
    
    # Set tick labels font (increased size for paper)
    for label in ax1.get_xticklabels():
        label.set_fontfamily('serif')
        label.set_fontsize(18)  # Increased for paper
    for label in ax1.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_fontsize(18)  # Increased for paper
    ax1.tick_params(axis='y', labelsize=18)  # Increased for paper
    
    # Normal layout (no extra bottom padding needed)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF for publication only
    output_path_pdf = output_dir / "sudoku_uniform_noise_comparison.pdf"
    fig.savefig(output_path_pdf, format='pdf', bbox_inches="tight", dpi=300)
    
    plt.close(fig)
    
    print(f"Comparison plot saved to: {output_path_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uniform_noise_only results across multiple checkpoints"
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
    
    # Create comparison plot
    output_dir = Path(args.output_dir).resolve()
    plot_comparison(results_dict, output_dir)
    
    print(f"\nComparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

