### Import Packages ###
import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .MeanVariancePlot import MeanVariancePlot

def process_all_results(results_dir, image_dir):

    print("--- Starting Full Analysis ---")

    ### Define Master Plotting Aesthetics for ALL possible strategies ###
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

    ### Define Metrics and Plot Types to Generate ###
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CC']
    plot_types = {
        'trace': None,
        'trace_relative_iGS': 'iGS'
    }

    ### Create All Necessary Output Directories ###
    for metric in metrics_to_plot:
        for plot_folder in plot_types.keys():
            os.makedirs(os.path.join(image_dir, metric, plot_folder, 'trace'), exist_ok=True)
            os.makedirs(os.path.join(image_dir, metric, plot_folder, 'variance'), exist_ok=True)
            
    ### Discover Datasets and Aggregate Results ###
    all_pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    dataset_basenames = sorted(list(set([f.split('_')[0] for f in all_pkl_files])))

    for data_name in dataset_basenames:
        print(f"\nAggregating and processing dataset: {data_name}...")

        ## Find all result files for this specific dataset using a pattern ##
        search_pattern = os.path.join(results_dir, f"{data_name}_*_seed_*.pkl")
        result_files_for_dataset = glob.glob(search_pattern)

        if not result_files_for_dataset:
            print(f"  > Warning: No result files found for {data_name}. Skipping.")
            continue

        ## AGGREGATION STEP ##
        with open(result_files_for_dataset[0], 'rb') as f:
            first_result = pickle.load(f)
        aggregated_results = {strategy: {metric: [] for metric in metrics_df.keys()} for strategy, metrics_df in first_result.items()}

        ## Loop through all files ##
        for i, file_path in enumerate(result_files_for_dataset):
            with open(file_path, 'rb') as f:
                single_run_result = pickle.load(f)
            for strategy, metrics_df in single_run_result.items():
                if strategy in aggregated_results: 
                    for metric, series in metrics_df.items():
                        if metric in aggregated_results[strategy]:
                            series.name = f"Sim_{i}"
                            aggregated_results[strategy][metric].append(series)
        
        ## Concatenate everything ##
        final_results_for_plotting = {}
        for strategy, metrics_dict in aggregated_results.items():
            final_results_for_plotting[strategy] = {}
            for metric, series_list in metrics_dict.items():
                if series_list: 
                    final_results_for_plotting[strategy][metric] = pd.concat(series_list, axis=1)

        ## Generate plots ##
        for metric in metrics_to_plot:
            results_for_metric = {
                strategy: data[metric] 
                for strategy, data in final_results_for_plotting.items() 
                if metric in data and not data[metric].empty
            }

            if not results_for_metric:
                print(f"  > Warning: No results found for metric '{metric}'. Skipping plot.")
                continue

            for folder_name, relative_error_baseline in plot_types.items():
                
                y_label = f"Normalized {metric}" if relative_error_baseline else metric
                subtitle = f"Performance ({metric}) on {data_name.upper()} Dataset"

                TracePlotMean, TracePlotVariance = MeanVariancePlot(
                    RelativeError=relative_error_baseline,
                    Colors=master_colors, LegendMapping=master_legend, Linestyles=master_linestyles,
                    Y_Label=y_label, Subtitle=subtitle,
                    TransparencyVal=0.1, VarInput=True, CriticalValue=1.96,
                    initial_train_proportion=0.16, candidate_pool_proportion=0.64,
                    **results_for_metric
                )

                # Save the plots #
                base_plot_path = os.path.join(image_dir, metric, folder_name)
                trace_plot_path = os.path.join(base_plot_path, 'trace', f"{data_name}_{metric}_TracePlot.png")
                TracePlotMean.savefig(trace_plot_path, bbox_inches='tight', dpi=300)
                plt.close(TracePlotMean)

                if TracePlotVariance:
                    variance_plot_path = os.path.join(base_plot_path, 'variance', f"{data_name}_{metric}_VariancePlot.png")
                    TracePlotVariance.savefig(variance_plot_path, bbox_inches='tight', dpi=300)
                    plt.close(TracePlotVariance)
        print(f"Saved all plots for {data_name}.")

    print("\n--- Analysis Complete ---")