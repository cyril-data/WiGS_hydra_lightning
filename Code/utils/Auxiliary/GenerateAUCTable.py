### LIBRARIES ####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

### CONFIGURATION ###
METRIC = 'RMSE'  # Options: 'RMSE', 'MAE', 'R2', 'CC'
BASELINE_METHOD = 'iGS'  # iGS, Random
OUTPUT_FILENAME = 'AUC_Performance_Heatmap'

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
                    # Priority 1: Use pre-calculated 'mean' column
                    if 'mean' in val.columns:
                        selector_means[selector] = val['mean']
                    elif 'Mean' in val.columns:
                        selector_means[selector] = val['Mean']
                    else:
                        # Priority 2: Calculate mean across all seed columns
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

def generate_heatmap(auc_df, output_dir):
    """
    Generates the relative performance heatmap.
    """
    if auc_df.empty:
        print("Error: No AUC data calculated.")
        return

    # 1. Pivot to get Matrix: Index=Dataset, Columns=Selector
    pivot_df = auc_df.pivot(index='Dataset', columns='Selector', values='AUC')
    
    # 2. Check if Baseline exists
    if BASELINE_METHOD not in pivot_df.columns:
        print(f"Error: Baseline method '{BASELINE_METHOD}' not found in data.")
        print(f"Available methods: {pivot_df.columns.tolist()}")
        return

    # 3. Calculate Ratio relative to Baseline (Method / Baseline)
    ratio_df = pivot_df.div(pivot_df[BASELINE_METHOD], axis=0)        
    plot_data = ratio_df.T
    
    #  REORDERING LOGIC
    plot_data = plot_data.sort_index(axis=1)
    
    # Sort Methods so BASELINE_METHOD is at the TOP, others alphabetical
    selectors = plot_data.index.tolist()
    if BASELINE_METHOD in selectors:
        selectors.remove(BASELINE_METHOD)
        selectors.sort()
        new_order = [BASELINE_METHOD] + selectors
        plot_data = plot_data.reindex(new_order)

    # 4. Plotting
    plt.figure(figsize=(24, 12))     
    cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
    
    sns.heatmap(plot_data, 
                     annot=True, 
                     fmt=".3f", 
                     cmap=cmap, 
                     center=1.0, 
                     vmin=0.90, vmax=1.10, 
                     linewidths=.5,
                     cbar_kws={'label': f'Relative AUC ({METRIC}) vs {BASELINE_METHOD}'})

    # plt.title(f'Active Learning Performance: Relative {METRIC} AUC vs {BASELINE_METHOD} (Lower is Better)', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Selection Strategy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 5. Save
    save_path_png = os.path.join(output_dir, f'{OUTPUT_FILENAME}.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to:\n  {save_path_png}")

def main():
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'images', 'manuscript')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Scanning {DATA_DIR}...")
    
    auc_df = load_and_calculate_auc_from_dirs(DATA_DIR)
    
    print("Generating Heatmap...")
    generate_heatmap(auc_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()