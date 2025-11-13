import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

def plot_average_weight_heatmap(dgp_name, selector, n_seeds, output_dir):
    """
    Generates a single heatmap where the color of each point is its
    AVERAGE selection weight across all N seeds.
    """
    print(f"--- Generating AVERAGE weight heatmap for: {dgp_name}, {selector} ---")
    print(f"--- Using {n_seeds} simulation seeds ---")

    # --- 1. Setup Paths and Parameters ---
    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    base_results_path = f"Results/simulation_results/aggregated/{dgp_name}"
    
    # Define all required file paths
    data_path = f"Data/processed/{dgp_name}.pkl"
    initial_path = f"{base_results_path}/InitialIndices.csv" # The shared initial indices
    selection_path = f"{base_results_path}/selection_history/{selector}_SelectionHistory.csv"
    weight_path = f"{base_results_path}/weight_history/{selector}_WeightHistory.csv"
    
    # Define output directory and file path
    plot_output_dir = os.path.join(output_dir, dgp_name, "weight_heatmaps")
    os.makedirs(plot_output_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_output_dir, f"{safe_selector_name}_AVERAGE_WeightHeatmap.png")

    # --- 2. Load All Data ---
    print("Loading data files...")
    try:
        df_data = pd.read_pickle(data_path)
        df_initial_full = pd.read_csv(initial_path)
        df_selection_full = pd.read_csv(selection_path)
        df_weight_full = pd.read_csv(weight_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        print(f"Missing file: {e.filename}")
        return

    # --- 3. Extract and Clean Data for ALL Seeds ---
    
    # Get the column names for the 50 seeds (e.g., "Sim_0", "Sim_1", ... "Sim_49")
    seed_cols = [f"Sim_{i}" for i in range(n_seeds)]
    
    # Check if we have enough data
    if not all(col in df_selection_full.columns for col in seed_cols):
        print(f"Error: Could not find all seed columns (Sim_0 to Sim_{n_seeds-1}) in {selection_path}")
        return
    if not all(col in df_weight_full.columns for col in seed_cols):
        print(f"Error: Could not find all seed columns in {weight_path}")
        return

    # --- Clean and Process Data ---
    print("Cleaning and processing data for all seeds...")
    
    # Get just the columns for our 50 seeds
    df_sel_seeds = df_selection_full[seed_cols]
    df_weight_seeds = df_weight_full[seed_cols]

    # --- Clean selection and weight data (fast method) ---
    def clean_df(df):
        # Stack all 50 columns into one giant series
        stacked_data = df.stack()
        # Clean it (strip brackets, convert to number, drop bad values)
        cleaned_data = pd.to_numeric(stacked_data.astype(str).str.strip('[]'), errors='coerce')
        # Unstack it back to its original shape (rows=iterations, cols=seeds)
        return cleaned_data.unstack()

    df_sel_cleaned = clean_df(df_sel_seeds)
    df_weight_cleaned = clean_df(df_weight_seeds)

    # --- "Melt" dataframes ---
    # This turns the 50 columns into two columns: 'seed_col' and 'value'
    # We get a long list of all selections and all weights
    sel_melted = df_sel_cleaned.melt(var_name='seed_col', value_name='index').dropna()
    weight_melted = df_weight_cleaned.melt(var_name='seed_col', value_name='weight').dropna()

    # --- 4. Calculate Average Weights ---
    print("Calculating average weights...")

    # Combine the lists by their index (they share Iteration and seed_col)
    df_merged = pd.concat([sel_melted['index'], weight_melted['weight']], axis=1).dropna()
    
    # --- This is the key step ---
    # Group by the point's index and calculate its mean weight
    avg_weights_by_index = df_merged.groupby('index')['weight'].mean()

    # --- 5. Prepare Data for Plotting ---
    
    # Get X/Y coordinates for all points
    df_plot = df_data[['X1', 'Y']].copy()
    
    # Join the average weight info to the main plot DataFrame
    df_plot = df_plot.join(avg_weights_by_index.rename('avg_weight'))
    
    # Get the set of all points that were in the initial set (across all 50 seeds)
    initial_indices_flat = df_initial_full[seed_cols].stack().dropna().astype(int).unique()
    initial_set = set(initial_indices_flat)
    
    # Get set of all points that were selected (and have an avg_weight)
    selected_set = set(avg_weights_by_index.index.astype(int))

    # Define the three groups for plotting
    df_plot['plot_type'] = 'unselected'
    df_plot.loc[list(initial_set), 'plot_type'] = 'initial'
    df_plot.loc[list(selected_set), 'plot_type'] = 'selected_avg' # Selected points override initial points
    
    # Get the separate data groups
    df_unselected = df_plot[df_plot['plot_type'] == 'unselected']
    df_initial_plot = df_plot[df_plot['plot_type'] == 'initial']
    df_selected_plot = df_plot[df_plot['plot_type'] == 'selected_avg']
    
    if df_selected_plot.empty:
        print("Note: No valid selected points found. Skipping heatmap.")
        return

    # --- 6. Create the Plot ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # 1. Plot all UNSELECTED points
    ax.scatter(df_unselected['X1'], df_unselected['Y'], 
               color='black', alpha=0.1, s=15)

    # 2. Plot the INITIAL points (gray, less fade)
    ax.scatter(df_initial_plot['X1'], df_initial_plot['Y'], 
               color='dimgray', alpha=0.8, s=25, label='Initial Set (any seed)', zorder=3)

    # 3. Plot the SELECTED points with the AVERAGE color gradient
    sc = ax.scatter(df_selected_plot['X1'], df_selected_plot['Y'], 
                    c=df_selected_plot['avg_weight'], 
                    cmap='coolwarm', # Blue (low) to Red (high)
                    s=25, 
                    label='Selected (Average)', 
                    vmin=0, vmax=1, 
                    zorder=2)

    # 4. Add the color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Average Selection Weight ($w_x$)', fontsize=12)

    # 5. Add DGP-specific annotations
    if dgp_name == "dgp_three_regime":
        for xpos in [0.4, 0.7, 0.6]:
            ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)

    # --- 7. Format and Save ---
    ax.set_title(f"Average Weight Distribution Heatmap: {selector}\n(Averaged across {n_seeds} seeds)", fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Average heatmap saved to: {output_plot_path}")
    print(f"--- Finished: {selector} (Average) ---")


def main():
    parser = argparse.ArgumentParser(description="Generate AVERAGE weight heatmap scatter plots.")
    
    parser.add_argument('--dgp_name', type=str, required=True,
                        help='Name of the data generating process (e.g., "dgp_three_regime")')
    parser.add_argument('--selector', type=str, required=True,
                        help='Name of the selector method (e.g., "WiGS (SAC)")')
    parser.add_argument('--n_seeds', type=int, required=True,
                        help='Number of seeds to average over (e.g., 50)')
    parser.add_argument('--output_dir', type=str, default="Results/visualizations",
                        help='Base directory to save the plots')
    
    args = parser.parse_args()
    
    plot_average_weight_heatmap(args.dgp_name, args.selector, args.n_seeds, args.output_dir)

if __name__ == "__main__":
    main()