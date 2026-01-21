### Libraries ###
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### CONFIGURATION ###
RESULTS_DIR = "/homes/simondn/WeightedGreedySampling/Results/simulation_results/aggregated/"
OUTPUT_DIR = "/homes/simondn/WeightedGreedySampling/Results/images/manuscript/"
TARGET_PERCENTAGES = [0.7, 0.8] 
BASELINE_NAME = "iGS"

### Mapping ###
NAME_MAPPING = {
    'Passive Learning': 'Random', 
    'GSx': 'GSx', 
    'GSy': 'GSy', 
    'iGS': 'iGS',
    'WiGS (Static w_x=0.75)': 'WiGS (Static, 0.75)',
    'WiGS (Static w_x=0.5)': 'WiGS (Static, 0.5)',
    'WiGS (Static w_x=0.25)': 'WiGS (Static, 0.25)', 
    'WiGS (Time-Decay, Linear)': 'WiGS (Linear)',
    'WiGS (Time-Decay, Exponential)': 'WiGS (Exp)',
    'WiGS (MAB-UCB1, c=0.5)': 'WiGS (MAB, c=0.5)', 
    'WiGS (MAB-UCB1, c=2.0)': 'WiGS (MAB, c=2.0)',
    'WiGS (MAB-UCB1, c=5.0)': 'WiGS (MAB, c=5.0)',
    'WiGS (SAC)': 'WiGS (SAC)',
    'QBC': 'QBC',
}

### Calculate relative N ###
def calculate_n_rel(results_dir):
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found.")
        return pd.DataFrame()

    datasets = sorted([d for d in os.listdir(results_dir) if not d.startswith(".")])
    print(f"Scanning {len(datasets)} datasets...")
    
    crossing_data = []

    for dataset in datasets:
        file_path = os.path.join(results_dir, dataset, "full_pool_metrics", "RMSE.pkl")
        
        if not os.path.exists(file_path):
            continue
            
        try:
            # 1. Load Data
            metrics_dict = pd.read_pickle(file_path)
            
            if not isinstance(metrics_dict, dict):
                continue

            # 2. Average across simulations
            mean_traces = {}
            for method_name, sim_df in metrics_dict.items():
                if isinstance(sim_df, pd.DataFrame):
                    mean_traces[method_name] = sim_df.mean(axis=1)
            
            df = pd.DataFrame(mean_traces)
            
            if BASELINE_NAME not in df.columns:
                continue

            # 3. Calculate Global Target
            rmse_min = df.min().min() # Best error achieved by ANY method
            rmse_start = df[BASELINE_NAME].iloc[0] # Starting error
            total_gap = rmse_start - rmse_min
            
            if total_gap < 1e-9: continue 

            # 4. Calculate N_rel for each target
            for k in TARGET_PERCENTAGES:
                target_rmse = rmse_start - (k * total_gap)
                
                # Baseline Crossing
                baseline_series = df[BASELINE_NAME]
                bl_crossings = baseline_series[baseline_series <= target_rmse].index
                n_baseline = float(bl_crossings[0]) if len(bl_crossings) > 0 else float(baseline_series.index[-1])
                if n_baseline == 0: n_baseline = 1.0

                # Method Crossings
                for method in df.columns:
                    if method not in NAME_MAPPING: continue
                    
                    method_series = df[method]
                    m_crossings = method_series[method_series <= target_rmse].index
                    n_method = float(m_crossings[0]) if len(m_crossings) > 0 else float(method_series.index[-1])
                    
                    n_rel = n_method / n_baseline
                    
                    crossing_data.append({
                        "Dataset": dataset,
                        "Method": NAME_MAPPING[method],
                        "Target": f"{int(k*100)}% Gain",
                        "N_rel": n_rel
                    })
                    
        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    return pd.DataFrame(crossing_data)

### Plotting ###
def plot_efficiency_boxplot(df):
    if df.empty:
        print("No data found to plot.")
        return
    sns.set_style("whitegrid")    
    order = [NAME_MAPPING[m] for m in NAME_MAPPING if m in NAME_MAPPING and NAME_MAPPING[m] in df["Method"].unique()]
    plt.figure(figsize=(10, 8)) 
    
    sns.boxplot(
        data=df,
        y="Method",
        x="N_rel",
        hue="Target",   
        order=order,
        orient="h",
        palette="viridis",
        showfliers=False,
        width=0.7
    )
    
    baseline_pretty = NAME_MAPPING.get(BASELINE_NAME, BASELINE_NAME)
    line_label = f"{baseline_pretty} Baseline (1.0)"
    plt.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label=line_label)    
    # targets_str = " & ".join([t.replace(" Gain", "") for t in df["Target"].unique()])
    # plt.title(f"Label Efficiency to Reach {targets_str} Performance Gains", fontsize=15, fontweight='bold')    
    plt.xlabel(f"Labels Needed Relative to {baseline_pretty} Baseline ($N_{{rel}}$)", fontsize=13)
    plt.ylabel("")
    plt.legend(title="Performance Target", loc="upper right")    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "DataEfficiency_Grouped.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
### Main ###
if __name__ == "__main__":
    print("Calculating relative efficiency...")
    df_ratios = calculate_n_rel(RESULTS_DIR)
    print(f"Processed {len(df_ratios)} data points.")
    plot_efficiency_boxplot(df_ratios)