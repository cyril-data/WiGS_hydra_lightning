import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_weight_heatmap(dgp_name, selector, seed, output_dir):
    """
    Generates a scatter plot for a single seed, where the color of each
    selected point is determined by its selection weight.
    """
    print(f"--- Generating weight heatmap for: {dgp_name}, {selector}, Seed: {seed} ---")

    # --- 1. Setup Paths and Parameters ---
    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    base_results_path = f"Results/simulation_results/aggregated/{dgp_name}"
    
    # Define all required file paths
    data_path = f"Data/processed/{dgp_name}.pkl"
    initial_path = f"{base_results_path}/InitialIndices.csv"
    selection_path = f"{base_results_path}/selection_history/{selector}_SelectionHistory.csv"
    weight_path = f"{base_results_path}/weight_history/{selector}_WeightHistory.csv"
    
    # Define output directory and file path
    plot_output_dir = os.path.join(output_dir, dgp_name, "weight_heatmaps")
    os.makedirs(plot_output_dir, exist_ok=True)
    output_plot_path = os.path.join(plot_output_dir, f"{safe_selector_name}_seed_{seed}_WeightHeatmap.png")

    # --- 2. Load All Data ---
    print("Loading data files...")
    try:
        df_data = pd.read_pickle(data_path)
        df_initial = pd.read_csv(initial_path)
        df_selection = pd.read_csv(selection_path)
        df_weight = pd.read_csv(weight_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        print(f"Missing file: {e.filename}")
        return

    # --- 3. Extract and Clean Data for Specific Seed ---
    seed_str = f"Sim_{seed}"
    
    # Check if seed exists in all files
    if seed_str not in df_initial.columns:
        print(f"Error: Column '{seed_str}' not found in {initial_path}")
        return
    if seed_str not in df_selection.columns:
        print(f"Error: Column '{seed_str}' not found in {selection_path}")
        return
    if seed_str not in df_weight.columns:
        print(f"Error: Column '{seed_str}' not found in {weight_path}")
        return

    # --- Clean and get all data lists ---
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

    # Find the shortest list to prevent errors
    num_selections = min(len(selection_indices), len(weights))
    selection_indices = selection_indices[:num_selections]
    weights = weights[:num_selections]

    # --- 4. Prepare Data for Plotting ---
    
    # Create a DataFrame of the selected points and their weights
    df_selected_points = pd.DataFrame({
        'index': selection_indices,
        'weight': weights
    }).set_index('index')

    # Get X/Y coordinates for all points
    df_plot = df_data[['X1', 'Y']].copy()
    df_plot['plot_type'] = 'unselected' # Start by marking all as 'unselected'
    df_plot.loc[initial_indices, 'plot_type'] = 'initial'
    
    # Join the weight info to the main plot DataFrame
    df_plot = df_plot.join(df_selected_points)
    
    # Mark selected points
    df_plot.loc[selection_indices, 'plot_type'] = 'selected'
    
    # Get the separate data groups for plotting
    df_unselected = df_plot[df_plot['plot_type'] == 'unselected']
    df_initial_plot = df_plot[df_plot['plot_type'] == 'initial']
    df_selected_plot = df_plot[df_plot['plot_type'] == 'selected']
    
    if df_selected_plot.empty:
        print("Note: No valid selected points with weights found. Skipping heatmap.")
        return

    # --- 5. Create the Plot ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # 1. Plot all UNSELECTED points
    ax.scatter(df_unselected['X1'], df_unselected['Y'], 
               color='gray', alpha=0.1, s=15, label='Unselected')

    # 2. Plot the INITIAL points
    ax.scatter(df_initial_plot['X1'], df_initial_plot['Y'], 
               color='dimgray', alpha=0.8, s=25, label='Initial Set', zorder=3)

    # 3. Plot the SELECTED points with the color gradient
    # We use vmin=0, vmax=1 to keep the color bar consistent
    sc = ax.scatter(df_selected_plot['X1'], df_selected_plot['Y'], 
                    c=df_selected_plot['weight'], 
                    cmap='coolwarm', # Blue (low) to Red (high)
                    s=25, 
                    label='Selected Points', 
                    vmin=0, vmax=1, 
                    zorder=2)

    # 4. Add the color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Selection Weight ($w_x$)', fontsize=12)

    # 5. Add DGP-specific annotations
    if dgp_name == "dgp_three_regime":
        for xpos in [0.4, 0.7, 0.6]:
            ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)

    # --- 6. Format and Save ---
    ax.set_title(f"Weight Distribution Heatmap: {selector}\nSeed: {seed}", fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"Heatmap saved to: {output_plot_path}")
    print(f"--- Finished: {selector}, Seed {seed} ---")


def main():
    parser = argparse.ArgumentParser(description="Generate weight heatmap scatter plots from simulation data.")
    
    parser.add_argument('--dgp_name', type=str, required=True,
                        help='Name of the data generating process (e.g., "dgp_three_regime")')
    parser.add_argument('--selector', type=str, required=True,
                        help='Name of the selector method (e.g., "WiGS (SAC)")')
    parser.add_argument('--seed', type=int, required=True,
                        help='Simulation seed to visualize (e.g., 0)')
    parser.add_argument('--output_dir', type=str, default="Results/visualizations",
                        help='Base directory to save the plots')
    
    args = parser.parse_args()
    
    plot_weight_heatmap(args.dgp_name, args.selector, args.seed, args.output_dir)

if __name__ == "__main__":
    main()