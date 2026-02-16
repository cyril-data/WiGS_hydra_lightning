### Import Packages ###
import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

### Plotting Function ###
def MeanVariancePlot(Subtitle=None,
                     TransparencyVal=0.2,
                     CriticalValue=1.96,
                     RelativeError=None,
                     Colors=None,
                     Linestyles=None,
                     xlim=None,
                     Y_Label=None,
                     VarInput=False,
                     initial_train_size: int = None,
                     FigSize=(9,12),
                     LegendMapping=None,
                     show_legend=True, 
                     **SimulationErrorResults):
    """
    Generates trace plots. 
    If RelativeError is provided, plots the DIFFERENCE (Method - Baseline).
    """
    if initial_train_size is None:
        raise ValueError("MeanVariancePlot requires 'initial_train_size' to be provided.")

    MeanVector, VarianceVector, StdErrorVector, StdErrorVarianceVector = {}, {}, {}, {}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis=1)
        VarianceVector[Label] = np.var(Results, axis=1)
        n_simulations = Results.shape[1]
        StdErrorVector[Label] = np.std(Results, axis=1) / np.sqrt(n_simulations)
        
        # Calculate Variance Bounds
        lower_chi2 = chi2.ppf(0.025, df=n_simulations - 1)
        upper_chi2 = chi2.ppf(0.975, df=n_simulations - 1)
        StdErrorVarianceVector[Label] = {
            "lower": (n_simulations - 1) * VarianceVector[Label] / upper_chi2,
            "upper": (n_simulations - 1) * VarianceVector[Label] / lower_chi2
        }

    ### Calculate Difference (Method - Baseline) if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = f"Error Difference (Method - {RelativeError})"
            
            BaselineMean = MeanVector[RelativeError].copy()
            
            for Label in MeanVector:
                MeanVector[Label] -= BaselineMean
                
                # Manual Clamp for 100% Labeled
                # At 100%, Method == Baseline, so Difference must be 0.0
                if len(MeanVector[Label]) > 0:
                    MeanVector[Label][-1] = 0.0
                    StdErrorVector[Label][-1] = 0.0
        else:
            print(f"  > Warning: Baseline '{RelativeError}' not found. Skipping normalization.")

    ### Mean Plot ###
    fig_mean, ax_mean = plt.subplots(figsize=FigSize)
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        num_iterations = len(MeanValues)
        total_pool_size = initial_train_size + num_iterations

        if num_iterations > 0:
            iterations_array = np.arange(num_iterations)
            num_labeled_at_step = initial_train_size + iterations_array
            x = (num_labeled_at_step / total_pool_size) * 100
        else:
            x = []
            
        color = Colors.get(Label, None) if Colors else None
        linestyle = Linestyles.get(Label, ':') if Linestyles else ':'
        legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
        
        ax_mean.plot(x, MeanValues, label=legend_label, color=color, linestyle=linestyle)
        ax_mean.fill_between(x, MeanValues - CriticalValue * StdErrorValues,
                             MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal, color=color)

    ax_mean.set_xlabel("Percent of Learning Pool Labeled")
    ax_mean.set_ylabel(Y_Label)
    
    # 3. Reference Line is now at 0.0 (No Difference)
    if RelativeError:
        ax_mean.axhline(y=0.0, color='r', linestyle='-', linewidth=1, alpha=0.5)

    if show_legend:
        ax_mean.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    if isinstance(xlim, list):
        ax_mean.set_xlim(xlim)


    ### Variance Plot ###
    fig_var = None
    if VarInput:
        fig_var, ax_var = plt.subplots(figsize=FigSize)
        for Label, VarianceValues in VarianceVector.items():
            num_iterations = len(VarianceValues)
            total_pool_size = initial_train_size + num_iterations

            if num_iterations > 0:
                iterations_array = np.arange(num_iterations)
                num_labeled_at_step = initial_train_size + iterations_array
                x = (num_labeled_at_step / total_pool_size) * 100
            else:
                x = []
            
            color = Colors.get(Label, None) if Colors else None
            linestyle = Linestyles.get(Label, '-') if Linestyles else '-'
            legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
            ax_var.plot(x, VarianceValues, label=legend_label, color=color, linestyle=linestyle)
            lower_bound = StdErrorVarianceVector[Label]["lower"]
            upper_bound = StdErrorVarianceVector[Label]["upper"]
            ax_var.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)
        
        ax_var.set_xlabel("Percent of Learning Pool Labeled")
        ax_var.set_ylabel("Variance of " + (Y_Label if Y_Label else "Error"))
        if show_legend:
            ax_var.legend(loc='upper right')
        if isinstance(xlim, list):
            ax_var.set_xlim(xlim)
    
    return (fig_mean, fig_var)
### Main Wrapper Function ###
def generate_all_plots(aggregated_results_dir, image_dir, show_legend=True, single_dataset=None):
    """
    Wrapper function to load aggregated .pkl files and generates all specified plots.
    Can process all datasets or just a single one if specified.
    """
    
    ### Aesthetics and Plot Definitions ###
    master_colors = {
        'Passive Learning': 'gray', 
        'GSx': 'cornflowerblue', 
        'GSy': 'salmon', 'iGS': 'red',
        'WiGS (Static w_x=0.75)': 'lightgreen', 
        'WiGS (Static w_x=0.25)': 'darkgreen', 
        'WiGS (Time-Decay, Linear)': 'orange',
        'WiGS (Time-Decay, Exponential)': 'saddlebrown', 
        'WiGS (MAB-UCB1, c=2.0)': 'darkviolet', 
        'WiGS (SAC)': 'darkcyan',
        'QBC': 'goldenrod'   
    }
    master_linestyles = {
        'Passive Learning': ':', 
        'GSx': ':', 
        'GSy': ':', 'iGS': '-',
        'WiGS (Static w_x=0.75)': '-.', 
        'WiGS (Static w_x=0.25)': '-.', 
        'WiGS (Time-Decay, Linear)': '-.',
        'WiGS (Time-Decay, Exponential)': '-.', 
        'WiGS (MAB-UCB1, c=2.0)': '-.', 
        'WiGS (SAC)': '-',
        'QBC': '-.' 
    }
    master_legend = {
        'Passive Learning': 'Random', 
        'GSx': 'GSx', 
        'GSy': 'GSy', 
        'iGS': 'iGS',
        'WiGS (Static w_x=0.75)': 'WiGS (Static, w_x=0.75)', 
        'WiGS (Static w_x=0.25)': 'WiGS (Static, w_x=0.25)', 
        'WiGS (Time-Decay, Linear)': 'WiGS (Linear Decay)',
        'WiGS (Time-Decay, Exponential)': 'WiGS (Exponential Decay)',
        'WiGS (MAB-UCB1, c=2.0)': 'MAB-UCB1, c=2.0',
        'WiGS (SAC)': 'WiGS (SAC)',
        'QBC': 'QBC',
    }
    
    ### Set up ###
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CC']
    plot_types = {'trace': None, 'trace_relative_iGS': 'iGS'}
    eval_types = ['full_pool']    
    strategies_to_exclude = {
    }


    ### Dynamically find datasets ###
    if single_dataset:
        dataset_folders = [single_dataset]
        print(f"--- Starting Plot Generation for single dataset: {single_dataset} ---")
    else:
        print("--- Starting Plot Generation from Aggregated Results ---")
        dataset_folders = [d for d in os.listdir(aggregated_results_dir) if os.path.isdir(os.path.join(aggregated_results_dir, d))]

    total_datasets = len(dataset_folders)
    

    for i, data_name in enumerate(dataset_folders):
        print(f"\n({i+1}/{total_datasets}) Processing dataset: {data_name}...")
        dataset_path = os.path.join(aggregated_results_dir, data_name)

        for eval_type in eval_types:
            print(f"  > Generating plots for '{eval_type}' metrics...")
            eval_metric_path = os.path.join(dataset_path, f"{eval_type}_metrics")

            if not os.path.isdir(eval_metric_path):
                print(f"    - Skipping: Directory not found at {eval_metric_path}")
                continue

            for metric in metrics_to_plot:
                metric_pkl_path = os.path.join(eval_metric_path, f"{metric}.pkl")
                
                if not os.path.exists(metric_pkl_path):
                    continue

                with open(metric_pkl_path, 'rb') as f:
                    results_for_metric = pickle.load(f)

                # Indices #                
                indices_file_path = os.path.join(dataset_path, 'InitialIndices.csv')
                try:
                    if not os.path.exists(indices_file_path):
                         raise FileNotFoundError 

                    indices_df = pd.read_csv(indices_file_path)
                    initial_train_size = len(indices_df)
                    if initial_train_size == 0:
                         raise ValueError("InitialIndices.csv is empty.") 
                except FileNotFoundError:
                    print(f"  > Warning: InitialIndices.csv not found for {data_name} at {indices_file_path}. Skipping {metric} plot.")
                    continue
                except ValueError as e:
                     print(f"  > Warning: Error reading InitialIndices.csv for {data_name}: {e}. Skipping {metric} plot.")
                     continue
                except Exception as e:
                     print(f"  > Warning: An unexpected error occurred loading InitialIndices.csv for {data_name}: {e}. Skipping {metric} plot.")
                     continue
                
                # Filter out the excluded strategies
                filtered_results = {strategy: df for strategy, df in results_for_metric.items() 
                                    if strategy not in strategies_to_exclude}
                
                for folder_name, baseline in plot_types.items():
                    y_label = f"Normalized {metric}" if baseline else metric
                    subtitle = f"Performance ({eval_type.capitalize()} {metric}) on {data_name.upper()} Dataset"

                    TracePlotMean, TracePlotVariance = MeanVariancePlot(RelativeError=baseline, 
                                                                        Colors=master_colors, 
                                                                        LegendMapping=master_legend, 
                                                                        Linestyles=master_linestyles, 
                                                                        Y_Label=y_label, 
                                                                        Subtitle=subtitle,
                                                                        TransparencyVal=0.1, 
                                                                        VarInput=True, 
                                                                        CriticalValue=1.96,
                                                                        initial_train_size=initial_train_size,
                                                                        show_legend=show_legend,
                                                                        **filtered_results)                    
                    output_eval_name = 'trace_plots' if eval_type == 'full_pool' else eval_type
                    base_plot_path = os.path.join(image_dir, output_eval_name, metric, folder_name)
                    os.makedirs(os.path.join(base_plot_path, 'trace'), exist_ok=True)
                    os.makedirs(os.path.join(base_plot_path, 'variance'), exist_ok=True)

                    trace_plot_path = os.path.join(base_plot_path, 'trace', f"{data_name}_{metric}_TracePlot.png")
                    TracePlotMean.savefig(trace_plot_path, bbox_inches='tight', dpi=300)
                    plt.close(TracePlotMean)

                    if TracePlotVariance:
                        variance_plot_path = os.path.join(base_plot_path, 'variance', f"{data_name}_{metric}_VariancePlot.png")
                        TracePlotVariance.savefig(variance_plot_path, bbox_inches='tight', dpi=300)
                        plt.close(TracePlotVariance)

        print(f"Finished all plots for {data_name}.")
    print("\n--- Plot Generation Complete ---")

### GENERATE LEGEND ###
def generate_legend(legend_mapping, colors, linestyles, output_path, ncol):
    """
    Generates a standalone legend image from the master style dictionaries.
    """
    
    # Create dummy plot handles for the legend
    handles = []
    labels = []
    
    for long_name, short_name in legend_mapping.items():
        color = colors.get(long_name)
        ls = linestyles.get(long_name, '-')
        
        if color is None:
            continue
            
        # Create a dummy line object
        line = plt.Line2D([0], [0], color=color, linestyle=ls, label=short_name)
        handles.append(line)
        labels.append(short_name)

    # Figure height needs to be slightly taller for 3 rows
    fig = plt.figure(figsize=(16, 3)) 
    
    # Create the legend
    fig_legend = fig.legend(
        handles, 
        labels, 
        loc='center', 
        frameon=True, 
        ncol=ncol 
    )    
    plt.gca().axis('off')    
    fig.savefig(
        output_path, 
        bbox_inches=fig_legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        dpi=300,
        transparent=True 
    )
    plt.close(fig)
    print(f"--- Legend generation complete ---")

### MAIN ###
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate plots for simulation results.")
    parser.add_argument('--dataset', type=str, required=False,  
                        help="Optional: name of a single dataset folder to process.")
    parser.add_argument('--no-legend', dest='show_legend', action='store_false',
                        help="Disable legends on individual plots (for later compilation).")    
    parser.add_argument('--legend_only', action='store_true',
                        help="If set, only generate a standalone legend file and exit.")
    args = parser.parse_args()

    ## Define Paths ##
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    AGGREGATED_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'Results', 'images')
    
    
    if args.legend_only:
        master_colors = {
            'Passive Learning': 'gray', 
            'GSx': 'cornflowerblue', 
            'GSy': 'salmon', 
            'iGS': 'red',
            'QBC': 'goldenrod',
            'Uncertainty Sampling': 'black',
            'EGAL': 'brown',
            'EMCM': 'teal',
            'WiGS (Static w_x=0.75)': 'lightgreen', 
            'WiGS (Static w_x=0.25)': 'darkgreen', 
            'WiGS (Time-Decay, Linear)': 'orange',
            'WiGS (Time-Decay, Exponential)': 'saddlebrown', 
            'WiGS (MAB-UCB1, c=2.0)': 'darkviolet', 
            'WiGS (SAC)': 'darkcyan'
        }

        master_linestyles = {
            'Passive Learning': ':', 
            'GSx': ':', 
            'GSy': ':', 
            'iGS': '-',
            'QBC': '-.' ,
            'Uncertainty Sampling': '--',
            'EGAL Density': '--',
            'EMCM': '--',
            'WiGS (Static w_x=0.75)': '-.', 
            'WiGS (Static w_x=0.25)': '-.', 
            'WiGS (Time-Decay, Linear)': '-.',
            'WiGS (Time-Decay, Exponential)': '-.', 
            'WiGS (MAB-UCB1, c=2.0)': '-.', 
            'WiGS (SAC)': '-'
        }

        master_legend = {
            'Passive Learning': 'Random', 
            'GSx': 'GSx', 
            'GSy': 'GSy', 
            'iGS': 'iGS',
            'QBC': 'QBC',
            'Uncertainty Sampling': 'Uncertainty Sampling',
            'EGAL': 'EGAL',
            'EMCM': 'EMCM',
            'WiGS (Static w_x=0.75)': 'WiGS (Static, w_x=0.75)',
            'WiGS (Static w_x=0.25)': 'WiGS (Static, w_x=0.25)', 
            'WiGS (Time-Decay, Linear)': 'WiGS (Linear Decay)',
            'WiGS (Time-Decay, Exponential)': 'WiGS (Exponential Decay)',
            'WiGS (MAB-UCB1, c=2.0)': 'WiGS (MAB, c=2.0)',
            'WiGS (SAC)': 'WiGS (SAC)'
        }
        
        # Define strategies to *exclude* from the legend #
        strategies_to_exclude = {
        }
        
        # Filter the master legend #
        filtered_legend_mapping = {
            long: short for long, short in master_legend.items() 
            if long not in strategies_to_exclude
        }

        # Define the output path #
        legend_output_path = os.path.join(IMAGE_DIR, "benchmark_legend.png")
        
        # Generate the legend #
        generate_legend(
            legend_mapping=filtered_legend_mapping,
            colors=master_colors,
            linestyles=master_linestyles,
            output_path=legend_output_path,
            ncol=7
        )

    else:
        ## Execute the main plotting function ##
        generate_all_plots(
            aggregated_results_dir=AGGREGATED_RESULTS_DIR,  
            image_dir=IMAGE_DIR,  
            show_legend=args.show_legend,
            single_dataset=args.dataset
        )