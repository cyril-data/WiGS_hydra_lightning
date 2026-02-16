import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

### CONFIGURATION ###
METRIC = 'RMSE'  # Options: 'RMSE', 'MAE', 'R2', 'CC'
TARGET_BASELINES = ['Passive Learning', 'iGS', 'QBC', 'WiGS (SAC)', None]

OUTPUT_FILENAME_BASE = 'AUC_Performance_Heatmap'

def load_and_calculate_auc_from_dirs(data_dir):
    """
    Traverses the directory structure:
      Results/aggregated/{DatasetName}/full_pool_metrics/{METRIC}.pkl
    
    Loads the metric DataFrame and calculates AUC for each selector.
    """
    auc_records = []
    
    # Get all dataset folders
    try:
        dataset_folders = sorted([
            f for f in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, f))
        ])
    except FileNotFoundError:
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    print(f"Found {len(dataset_folders)} dataset folders. Processing...")

    for dataset_name in dataset_folders:
        metric_path = os.path.join(data_dir, dataset_name, 'full_pool_metrics', f'{METRIC}.pkl')
        
        if not os.path.exists(metric_path):
            print(f"  [Skipping] {dataset_name}: {METRIC}.pkl not found.")
            continue
            
        try:
            # Load Data
            data = pd.read_pickle(metric_path)
            if not isinstance(data, dict):
                 print(f"  [Warning] {dataset_name}: Expected dictionary, got {type(data)}. Skipping.")
                 continue

            # Extract the mean curve for every selector
            selector_means = {}
            
            for selector, val in data.items():
                if isinstance(val, pd.DataFrame):
                    if 'mean' in val.columns:
                        selector_means[selector] = val['mean']
                    elif 'Mean' in val.columns:
                        selector_means[selector] = val['Mean']
                    else:
                        selector_means[selector] = val.mean(axis=1)
                        
                elif isinstance(val, pd.Series):
                    selector_means[selector] = val
            
            if not selector_means:
                print(f"  [Warning] {dataset_name}: Could not extract any data.")
                continue
                
            df = pd.DataFrame(selector_means)

            # AUC CALCULATION 
            if not np.issubdtype(df.index.dtype, np.number):
                 df.index = np.arange(len(df))
                 
            x = df.index.values
            
            for selector in df.columns:
                y = df[selector].values
                
                # Handle NaNs (Interpolate)
                if np.isnan(y).any():
                    y = pd.Series(y).interpolate().fillna(method='bfill').values
                
                # Integration (Trapezoidal rule)
                if hasattr(np, 'trapezoid'):
                     auc = np.trapezoid(y, x)
                else:
                     auc = np.trapz(y, x)
                
                auc_records.append({
                    'Dataset': dataset_name,
                    'Selector': selector,
                    'AUC': auc
                })
                
        except Exception as e:
            print(f"  [Error] {dataset_name}: {e}")

    return pd.DataFrame(auc_records)

def generate_heatmap(auc_df, output_dir, baseline_method):
    """
    Generates the performance heatmap. 
    If baseline_method is None, plots Absolute values.
    If baseline_method is a string, plots Relative ratio.
    """
    if auc_df.empty:
        print("Error: No AUC data calculated.")
        return

    # 1. Pivot to get Matrix: Index=Dataset, Columns=Selector
    pivot_df = auc_df.pivot(index='Dataset', columns='Selector', values='AUC')
    
    # Setup for Plotting
    plt.figure(figsize=(24, 12))     
    
    if baseline_method is None:
        print(f"\n--- Generating Heatmap (Absolute AUC) ---")
        plot_data = pivot_df.T
        plot_data = plot_data.sort_index(axis=1) 
        cmap = "viridis_r" 
        
        # Plot
        sns.heatmap(plot_data, 
                         annot=True, 
                         fmt=".1f", 
                         cmap=cmap, 
                         linewidths=.5,
                         cbar_kws={'label': f'Absolute Total AUC ({METRIC})'})
        
        title_str = f'Active Learning Performance: Absolute Total {METRIC} AUC (Lower is Better)'
        filename = f"{OUTPUT_FILENAME_BASE}_Absolute.png"

    else:
        # --- RELATIVE BASELINE MODE ---
        print(f"\n--- Generating Heatmap vs {baseline_method} ---")
        
        if baseline_method not in pivot_df.columns:
            print(f"  [Warning] Baseline method '{baseline_method}' not found in data.")
            return

        # Calculate Ratio
        ratio_df = pivot_df.div(pivot_df[baseline_method], axis=0)        
        plot_data = ratio_df.T
        plot_data = plot_data.sort_index(axis=1)
        
        # Reorder to put baseline on top
        selectors = plot_data.index.tolist()
        if baseline_method in selectors:
            selectors.remove(baseline_method)
            selectors.sort()
            new_order = [baseline_method] + selectors
            plot_data = plot_data.reindex(new_order)

        # Use Diverging colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
        
        # Plot
        sns.heatmap(plot_data, 
                         annot=True, 
                         fmt=".3f", 
                         cmap=cmap, 
                         center=1.0, 
                         vmin=0.90, vmax=1.10, 
                         linewidths=.5,
                         cbar_kws={'label': f'Relative AUC ({METRIC}) vs {baseline_method}'})

        title_str = f'Active Learning Performance: Relative {METRIC} AUC vs {baseline_method} (Lower is Better)'
        safe_baseline = baseline_method.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{OUTPUT_FILENAME_BASE}_vs_{safe_baseline}.png"

    # Common Plot Settings
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Selection Strategy', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    # plt.title(title_str, fontsize=16)
    plt.tight_layout()
    
    # Save
    save_path_png = os.path.join(output_dir, filename)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {save_path_png}")

def main():
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'images', 'manuscript', 'AUC_Tables')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Scanning {DATA_DIR}...")
    
    auc_df = load_and_calculate_auc_from_dirs(DATA_DIR)    
    for baseline in TARGET_BASELINES:
        generate_heatmap(auc_df, OUTPUT_DIR, baseline)
    
    print("\nAll heatmaps generated.")

if __name__ == "__main__":
    main()