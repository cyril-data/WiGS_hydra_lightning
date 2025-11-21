### LIBRARIES ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm

### SINGLE HEATMAP FUNCTION ###
def plot_single_seed_heatmap(dgp_name, selector, seed, output_dir, df_data, df_initial, df_selection, df_weight):
    """
    Generates a scatter plot for a single seed (EXHAUSTIVE version).
    Assumes all points are either initial or selected.
    """
    print(f"--- Generating SINGLE SEED weight heatmap for: {dgp_name}, {selector}, Seed: {seed} ---")
    
    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    plot_output_dir = os.path.join(output_dir, dgp_name, "weight_heatmaps")
    os.makedirs(plot_output_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_output_dir, f"{safe_selector_name}_seed_{seed}_WeightHeatmap.png")

    ## Extract and Clean Data for Specific Seed ##
    seed_str = f"Sim_{seed}"
    if seed_str not in df_initial.columns:
        print(f"Error: Column '{seed_str}' not found in InitialIndices.csv")
        return
    if seed_str not in df_selection.columns:
        print(f"Error: Column '{seed_str}' not found in selection_history.csv")
        return
    if seed_str not in df_weight.columns:
        print(f"Error: Column '{seed_str}' not found in weight_history.csv")
        return

    ## Clean and get all data lists ##
    initial_indices = (
        df_initial[seed_str].dropna().astype(str)
        .str.strip('[]').astype(float).astype(int).tolist()
    )
    selection_indices = (
        df_selection[seed_str].dropna().astype(str)
        .str.strip('[]').astype(float).astype(int).tolist()
    )
    weights = (
        pd.to_numeric(
            df_weight[seed_str].astype(str).str.strip('[]'), 
            errors='coerce'
        ).dropna().tolist()
    )

    num_selections = min(len(selection_indices), len(weights))
    selection_indices = selection_indices[:num_selections]
    weights = weights[:num_selections]

    ## Prepare Data for Plotting ##
    df_selected_points = pd.DataFrame({
        'index': selection_indices, 'weight': weights
    }).set_index('index')

    df_plot = df_data[['X1', 'Y']].copy()
    df_plot = df_plot.join(df_selected_points)
    
    # --- MODIFIED LOGIC ---
    df_plot['plot_type'] = 'selected'
    df_plot.loc[initial_indices, 'plot_type'] = 'initial'
    df_plot.loc[selection_indices, 'plot_type'] = 'selected'    
    df_initial_plot = df_plot[df_plot['plot_type'] == 'initial']
    df_selected_plot = df_plot[df_plot['plot_type'] == 'selected']
    
    if df_selected_plot.empty:
        print("Note: No valid selected points with weights found. Skipping heatmap.")
        return

    # Create the Plot #
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.scatter(df_initial_plot['X1'], df_initial_plot['Y'], 
               color='dimgray', alpha=0.8, s=25, label='Initial Set', zorder=3)
    sc = ax.scatter(df_selected_plot['X1'], df_selected_plot['Y'], 
                    c=df_selected_plot['weight'], cmap='coolwarm', 
                    s=25, label='Selected', vmin=0, vmax=1, zorder=2)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Selection Weight ($w_x$)', fontsize=12)

    if dgp_name == "dgp_three_regime":
        for xpos in [0.4, 0.7, 0.6]:
            ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)

    # Format and Save #
    ax.set_title(f"Weight Distribution Heatmap: {selector}\nSeed: {seed}", fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Single-seed heatmap saved to: {output_plot_path}")

### AVERAGE HEATMAP FUNCTION ###
def plot_average_heatmap(dgp_name, selector, output_dir, df_data, df_initial_full, df_selection_full, df_weight_full):
    """
    Generates a single heatmap of the AVERAGE selection weight (EXHAUSTIVE version).
    Assumes all points are either initial or selected across all seeds.
    """
    print(f"--- Generating AVERAGE weight heatmap for: {dgp_name}, {selector} ---")

    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    output_plot_path = os.path.join(output_dir, f"{dgp_name}_{safe_selector_name}_AVERAGE_WeightHeatmap.png")

    ## Extract and Clean Data for ALL Seeds ##
    seed_cols = [col for col in df_selection_full.columns if col.startswith('Sim_')]
    if not seed_cols:
        print(f"Error: No simulation columns (Sim_X) found in selection_history.csv")
        return
    if not all(col in df_weight_full.columns for col in seed_cols):
        print(f"Error: Mismatch in seed columns between selection and weight files.")
        return
        
    n_seeds = len(seed_cols)
    print(f"--- Dynamically found {n_seeds} simulation seeds ---")
    
    print("Cleaning and processing data for all seeds...")
    df_sel_seeds = df_selection_full[seed_cols]
    df_weight_seeds = df_weight_full[seed_cols]

    def clean_df(df):
        stacked_data = df.stack()
        cleaned_data = pd.to_numeric(stacked_data.astype(str).str.strip('[]'), errors='coerce')
        return cleaned_data.unstack()

    df_sel_cleaned = clean_df(df_sel_seeds)
    df_weight_cleaned = clean_df(df_weight_seeds)

    sel_melted = df_sel_cleaned.melt(var_name='seed_col', value_name='index').dropna()
    weight_melted = df_weight_cleaned.melt(var_name='seed_col', value_name='weight').dropna()

    ## Calculate Average Weights ##
    print("Calculating average weights...")
    df_merged = pd.concat([sel_melted['index'], weight_melted['weight']], axis=1).dropna()
    avg_weights_by_index = df_merged.groupby('index')['weight'].mean()

    ## Prepare Data for Plotting ##
    df_plot = df_data[['X1', 'Y']].copy()
    df_plot = df_plot.join(avg_weights_by_index.rename('avg_weight'))
    
    initial_indices_flat = df_initial_full[seed_cols].stack().dropna().astype(int).unique()
    
    initial_set = set(initial_indices_flat)
    selected_set = set(avg_weights_by_index.index.astype(int))

    df_plot['plot_type'] = 'selected_avg'
    
    only_initial_indices = list(initial_set - selected_set)
    df_plot.loc[only_initial_indices, 'plot_type'] = 'initial'

    df_initial_plot = df_plot[df_plot['plot_type'] == 'initial']
    df_selected_plot = df_plot[df_plot['plot_type'] == 'selected_avg']
    
    if df_selected_plot.empty:
        print("Note: No valid selected points found. Skipping heatmap.")
        return

    ## Create the Plot ##
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    if not df_initial_plot.empty:
        print(f"--- Found {len(df_initial_plot)} points that were ONLY initial (never selected) ---")
        ax.scatter(df_initial_plot['X1'], df_initial_plot['Y'],
                   color='purple', alpha=0.8, s=50, 
                   label='Initial Set Only (Never Selected)', zorder=3)

    sc = ax.scatter(df_selected_plot['X1'], 
                    df_selected_plot['Y'],
                    c=df_selected_plot['avg_weight'], 
                    cmap='coolwarm',
                    s=25, 
                    label='Selected (Average)', 
                    vmin=0, 
                    vmax=1, 
                    zorder=2)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Average Selection Weight ($w_x$)', fontsize=12)

    if dgp_name == "dgp_three_regime":
        for xpos in [0.4, 0.7, 0.6]:
            ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)

    ## Format and Save ##
    # ax.set_title(f"Average Weight Distribution Heatmap: {selector}\n(Averaged across {n_seeds} seeds)", fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    # ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Average heatmap saved to: {output_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate single-seed or average weight heatmap plots.")
    
    parser.add_argument('--dgp_name', type=str, required=True,
                        help='Name of the data generating process (e.g., "dgp_three_regime")')
    parser.add_argument('--selector', type=str, required=True,
                        help='Name of the selector method (e.g., "WiGS (SAC)")')
    parser.add_argument('--seed', type=str, required=True,
                        help='Simulation seed to visualize (e.g., "0") or "avg" for the average plot')
    parser.add_argument('--output_dir', type=str, default="Results/visualizations",
                        help='Base directory to save the plots')
    
    args = parser.parse_args()

    ### 1. Define Paths and Load Shared Data ###
    base_results_path = f"Results/simulation_results/aggregated/{args.dgp_name}"
    data_path = f"Data/processed/{args.dgp_name}.pkl"
    initial_path = f"{base_results_path}/InitialIndices.csv"
    selection_path = f"{base_results_path}/selection_history/{args.selector}_SelectionHistory.csv"
    weight_path = f"{base_results_path}/weight_history/{args.selector}_WeightHistory.csv"

    print("Loading all data files...")
    try:
        df_data = pd.read_pickle(data_path)
        x_min = df_data['X1'].min()
        x_max = df_data['X1'].max()
        df_data['X1'] = (df_data['X1'] - x_min) / (x_max - x_min)
        df_initial_full = pd.read_csv(initial_path)
        df_selection_full = pd.read_csv(selection_path)
        df_weight_full = pd.read_csv(weight_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}", file=sys.stderr)
        print(f"Missing file: {e.filename}", file=sys.stderr)
        sys.exit(1)
    
    ### 2. Route to the correct function ###
    if args.seed.lower() == 'avg' or args.seed.lower() == 'all':
        plot_average_heatmap(args.dgp_name, args.selector, args.output_dir,
                             df_data, df_initial_full, df_selection_full, df_weight_full)
    else:
        try:
            seed_int = int(args.seed)
            plot_single_seed_heatmap(args.dgp_name, args.selector, seed_int, args.output_dir,
                                     df_data, df_initial_full, df_selection_full, df_weight_full)
        except ValueError:
            print(f"Error: --seed must be an integer (like '0') or the string 'avg'. You provided: {args.seed}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()