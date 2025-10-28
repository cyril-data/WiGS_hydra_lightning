import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm 

def plot_weight_trends(dgp_name, selector, seed_to_plot, output_dir):
    """
    Generates a trend graph of the w_x weight for a given selector.
    Plots either a single seed's trend or the average trend across all seeds.
    """
    # --- 1. Define Paths and Load Data ---
    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    weight_file_path = f"Results/simulation_results/aggregated/{dgp_name}/weight_history/{selector}_WeightHistory.csv"
    plot_output_dir = os.path.join(output_dir, dgp_name, "weight_trends")
    os.makedirs(plot_output_dir, exist_ok=True)

    try:
        df_full = pd.read_csv(weight_file_path, index_col='Iteration') 
    except FileNotFoundError:
        print(f"Error: Could not find file: {weight_file_path}")
        return
    except Exception as e:
        print(f"Error loading {weight_file_path}: {e}")
        return

    # --- 2. Determine Plotting Mode (Single Seed or Aggregate) ---
    plot_single_seed = False
    target_seed = None
    if seed_to_plot.lower() != "all":
        try:
            target_seed = int(seed_to_plot)
            seed_str = f"Sim_{target_seed}"
            if seed_str not in df_full.columns:
                print(f"Error: Column '{seed_str}' not found in the file.")
                return
            plot_single_seed = True
        except ValueError:
            print(f"Error: --seed must be an integer or 'all'. Received '{seed_to_plot}'.")
            return

    print(f"--- Analyzing weight trend for: {dgp_name}, {selector}, Seed(s): {seed_to_plot} ---")

    # --- 3. Clean and Process Data ---
    sim_columns = [col for col in df_full.columns if col.startswith('Sim_')]
    if not sim_columns:
        print(f"Error: No simulation columns (Sim_X) found in {weight_file_path}")
        return

    # Clean all simulation columns simultaneously
    df_cleaned = df_full[sim_columns].copy()
    for col in sim_columns:
        temp_col = df_cleaned[col].astype(str).str.strip('[]')
        df_cleaned[col] = pd.to_numeric(temp_col, errors='coerce')

    # --- 4. Prepare Data for Plotting ---
    x_axis = df_cleaned.index 
    if plot_single_seed:
        weights_to_plot = df_cleaned[seed_str]
        plot_title = f"Weight Trend: {selector}\nDataset: {dgp_name} - Seed: {target_seed}"
        output_filename = f"{safe_selector_name}_seed_{target_seed}_WeightTrend.png"
        if weights_to_plot.isnull().all():
            print("Note: No valid weight data found for this seed. Skipping plot.")
            return
    else: 
        weights_mean = df_cleaned[sim_columns].mean(axis=1)
        weights_std = df_cleaned[sim_columns].std(axis=1)
        weights_upper = weights_mean + weights_std
        weights_lower = weights_mean - weights_std
        weights_upper = weights_upper.clip(0, 1)
        weights_lower = weights_lower.clip(0, 1)

        plot_title = f"Average Weight Trend: {selector}\nDataset: {dgp_name} (Avg across {len(sim_columns)} seeds)"
        output_filename = f"{safe_selector_name}_all_seeds_AvgWeightTrend.png"
        if weights_mean.isnull().all():
             print("Note: No valid weight data found across any seeds. Skipping plot.")
             return

    output_plot_path = os.path.join(plot_output_dir, output_filename)

    # --- 5. Plot the Graph ---
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    if plot_single_seed:
        ax.plot(x_axis, weights_to_plot, label=f"Weight Trend (Seed {target_seed})", color="#0033A0", linewidth=1.5)
    else:
        ax.plot(x_axis, weights_mean, label=f"Average Weight (±1 Std Dev)", color="#0033A0", linewidth=2)
        ax.fill_between(x_axis, weights_lower, weights_upper, color="#0033A0", alpha=0.2)

    # --- 6. Format and Save ---
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Weight ($w_x$)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-0.05, 1.05) 
    ax.set_xlim(x_axis.min(), x_axis.max())
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300) 
    plt.close()

    print(f"Trend graph saved to: {output_plot_path}")
    print(f"--- Finished: {selector}, Seed(s): {seed_to_plot} ---")


def main():
    parser = argparse.ArgumentParser(description="Generate weight trend graphs from simulation data.")
    parser.add_argument('--dgp_name', type=str, required=True,
                        help='Name of the data generating process (e.g., "dgp_three_regime")')
    parser.add_argument('--selector', type=str, required=True,
                        help='Name of the selector method (e.g., "WiGS (SAC)")')
    parser.add_argument('--seed', type=str, required=True,
                        help='Simulation seed to visualize (e.g., 0) or "all" for average')
    parser.add_argument('--output_dir', type=str, default="Results/visualizations",
                        help='Base directory relative to project root to save the plots')

    args = parser.parse_args()

    # --- Define Project Root Dynamically (within main scope) ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except NameError:
        project_root = os.getcwd()
    if not os.path.isabs(args.output_dir):
         absolute_output_dir = os.path.join(project_root, args.output_dir)
    else:
         absolute_output_dir = args.output_dir

    plot_weight_trends(args.dgp_name, args.selector, args.seed, absolute_output_dir)

if __name__ == "__main__":
    main()