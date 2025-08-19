### Import Packages ###
import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Make sure you have this import if the function is in another file
from .MeanVariancePlot import MeanVariancePlot

def generate_all_plots(aggregated_results_dir, image_dir):
    """
    Loads aggregated .pkl files and generates all specified plots.
    """
    print("--- Starting Plot Generation from Aggregated Results ---")
    
    # --- Aesthetics and Plot Definitions (no changes here) ---
    master_colors = {
        'Passive Learning': 'gray', 'GSx': 'cornflowerblue', 'GSy': 'salmon', 'iGS': 'red',
        'WiGS (Static w_x=0.75)': 'lightgreen', 'WiGS (Static w_x=0.5)': 'forestgreen',
        'WiGS (Static w_x=0.25)': 'darkgreen', 'WiGS (Time-Decay, Linear)': 'orange',
        'WiGS (Time-Decay, Exponential)': 'saddlebrown', 'WiGS (MAB-UCB1, c=0.5)': 'orchid',
        'WiGS (MAB-UCB1, c=2.0)': 'darkviolet', 'WiGS (MAB-UCB1, c=5.0)': 'indigo'
    }
    master_linestyles = {
        'Passive Learning': ':', 'GSx': ':', 'GSy': ':', 'iGS': '-',
        'WiGS (Static w_x=0.75)': '-', 'WiGS (Static w_x=0.5)': '-.',
        'WiGS (Static w_x=0.25)': '--', 'WiGS (Time-Decay, Linear)': '-',
        'WiGS (Time-Decay, Exponential)': '-.', 'WiGS (MAB-UCB1, c=0.5)': '-',
        'WiGS (MAB-UCB1, c=2.0)': '-', 'WiGS (MAB-UCB1, c=5.0)': '-'
    }
    master_legend = {
        'Passive Learning': 'Random', 'GSx': 'GSx', 'GSy': 'GSy', 'iGS': 'iGS',
        'WiGS (Static w_x=0.75)': 'WiGS (Static, w_x=0.75)', 'WiGS (Static w_x=0.5)': 'WiGS (Static, w_x=0.5)',
        'WiGS (Static w_x=0.25)': 'WiGS (Static, w_x=0.25)', 'WiGS (Time-Decay, Linear)': 'WiGS (Linear Decay)',
        'WiGS (Time-Decay, Exponential)': 'WiGS (Exponential Decay)', 'WiGS (MAB-UCB1, c=0.5)': 'WiGS (MAB, c=0.5)',
        'WiGS (MAB-UCB1, c=2.0)': 'WiGS (MAB, c=2.0)', 'WiGS (MAB-UCB1, c=5.0)': 'WiGS (MAB, c=5.0)'
    }
    
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CC']
    plot_types = {'trace': None, 'trace_relative_iGS': 'iGS'}

    for metric in metrics_to_plot:
        for plot_folder in plot_types.keys():
            os.makedirs(os.path.join(image_dir, metric, plot_folder), exist_ok=True)
            
    dataset_folders = [d for d in os.listdir(aggregated_results_dir) if os.path.isdir(os.path.join(aggregated_results_dir, d))]

    # --- RESTRUCTURED LOOP LOGIC ---
    for data_name in dataset_folders:
        print(f"\nProcessing dataset: {data_name}...")
        dataset_path = os.path.join(aggregated_results_dir, data_name)

        for metric in metrics_to_plot:
            # 1. Construct the path to the specific metric file
            metric_pkl_path = os.path.join(dataset_path, f"{metric}.pkl")

            # 2. Check if the file exists and load it
            if not os.path.exists(metric_pkl_path):
                print(f"  > Warning: File '{metric}.pkl' not found for dataset '{data_name}'. Skipping.")
                continue
            
            with open(metric_pkl_path, 'rb') as f:
                results_for_metric = pickle.load(f)
            
            print(f"  > Plotting metric: {metric}")

            # 3. Generate the plots for the loaded metric data
            for folder_name, relative_error_baseline in plot_types.items():
                y_label = f"Normalized {metric}" if relative_error_baseline else metric
                subtitle = f"Performance ({metric}) on {data_name.upper()} Dataset"

                # Ensure baseline exists before trying to normalize
                if relative_error_baseline and relative_error_baseline not in results_for_metric:
                    print(f"  > Warning: Baseline '{relative_error_baseline}' not in results for {metric}. Skipping relative plot.")
                    continue

                TracePlotMean, TracePlotVariance = MeanVariancePlot(
                    RelativeError=relative_error_baseline,
                    Colors=master_colors, LegendMapping=master_legend, Linestyles=master_linestyles,
                    Y_Label=y_label, Subtitle=subtitle,
                    TransparencyVal=0, VarInput=True, CriticalValue=1.96, # Increased transparency a bit
                    initial_train_proportion=0.16, candidate_pool_proportion=0.64,
                    **results_for_metric
                )

                base_plot_path = os.path.join(image_dir, metric, folder_name)
                trace_plot_path = os.path.join(base_plot_path, f"{data_name}_{metric}_TracePlot.png")
                TracePlotMean.savefig(trace_plot_path, bbox_inches='tight', dpi=300)
                plt.close(TracePlotMean)

                # if TracePlotVariance:
                #     variance_plot_path = os.path.join(base_plot_path, f"{data_name}_{metric}_VariancePlot.png")
                #     TracePlotVariance.savefig(variance_plot_path, bbox_inches='tight', dpi=300)
                #     plt.close(TracePlotVariance)
        
        print(f"Finished all plots for {data_name}.")

    print("\n--- Plot Generation Complete ---")