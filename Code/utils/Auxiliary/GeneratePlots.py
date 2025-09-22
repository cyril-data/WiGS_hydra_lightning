### Import Packages ###
import os
import pickle
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

### Import packages ###
from scipy.stats import chi2

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
                     FigSize=(10, 4),
                     LegendMapping=None,
                     initial_train_proportion=0.16,
                     candidate_pool_proportion=0.64,
                     **SimulationErrorResults):
    """
    Generates and returns trace plots for the mean and variance of simulation results.

    Args:
        Subtitle (str): A subtitle to display on the plot(s).
        TransparencyVal (float): The alpha transparency for the confidence interval shading.
        CriticalValue (float): The z-score for the confidence interval.
        RelativeError (str): The name of a strategy in SimulationErrorResults to use as a baseline for normalization. 
            If provided, all other curves will be divided by this baseline.
        Colors (Dict[str, str]): A dictionary mapping strategy names to matplotlib color strings. 
        Linestyles (Dict[str, str]): A dictionary mapping strategy names to matplotlib linestyle strings. 
        xlim (list): A list of two numbers [min, max] to set the x-axis limits.
        Y_Label (str): The label for the y-axis. 
        VarInput (bool): If True, a second plot for the variance of the error will also be generated.
        FigSize (Tuple[int, int]): The dimensions of the plot figures.
        LegendMapping (Dict[str, str]): A dictionary to map strategy names to more descriptive labels for the plot legend. 
        initial_train_proportion (float): The proportion of the dataset used for the initial training set, used to calculate the x-axis.
            This is 0.16 when the `TEST_PROPORTION` and `CANDIDATE_PROPORTION` input in `CreateSimulationSBatch.py` are 
            0.2 and 0.8 respectively. 
        candidate_pool_proportion (float): The proportion of the dataset in the candidate pool, used to calculate the x-axis.
            This is 0.64 when the `TEST_PROPORTION` and `CANDIDATE_PROPORTION` input in `Code/Cluster/CreateSimulationSBatch.py` 
            are 0.2 and 0.8 respectively. 
        **SimulationErrorResults (Dict[str, pd.DataFrame]): Arbitrary keyword arguments
            where each key is a strategy name and the value is a pandas DataFrame
            of its results. Rows should be iterations and columns should be
            simulation runs.

    Returns:
        Tuple[plt.Figure, Optional[plt.Figure]]: A tuple containing:
            - fig_mean (plt.Figure): The Matplotlib figure object for the mean trace plot.
            - fig_var (plt.Figure or None): The Matplotlib figure object for the
              variance trace plot if VarInput is True, otherwise None.
    """


    ### Set Up ###
    MeanVector, VarianceVector, StdErrorVector, StdErrorVarianceVector = {}, {}, {}, {}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis=1)
        VarianceVector[Label] = np.var(Results, axis=1)
        n_simulations = Results.shape[1]
        StdErrorVector[Label] = np.std(Results, axis=1) / np.sqrt(n_simulations)
        lower_chi2 = chi2.ppf(0.025, df=n_simulations - 1)
        upper_chi2 = chi2.ppf(0.975, df=n_simulations - 1)
        StdErrorVarianceVector[Label] = {
            "lower": (n_simulations - 1) * VarianceVector[Label] / upper_chi2,
            "upper": (n_simulations - 1) * VarianceVector[Label] / lower_chi2
        }

    ### Normalize to Relative Error if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = f"Normalized Error (Baseline: {RelativeError}=1.0)"
            BaselineMean = MeanVector[RelativeError].copy()
            BaselineVariance = VarianceVector[RelativeError].copy()
            for Label in MeanVector:
                MeanVector[Label] /= BaselineMean
                StdErrorVector[Label] /= BaselineMean
                VarianceVector[Label] /= BaselineVariance
        else:
            raise ValueError(f"RelativeError='{RelativeError}' not found in provided results.")

    ### Mean Plot ###
    fig_mean, ax_mean = plt.subplots(figsize=FigSize)
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        num_iterations = len(MeanValues)
        if num_iterations > 1:
            iterations_array = np.arange(num_iterations)
            x = (initial_train_proportion + (iterations_array / (num_iterations - 1)) * candidate_pool_proportion) * 100
        else:
            x = [initial_train_proportion * 100]
        color = Colors.get(Label, None) if Colors else None
        linestyle = Linestyles.get(Label, ':') if Linestyles else ':'
        legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
        
        ax_mean.plot(x, MeanValues, label=legend_label, color=color, linestyle=linestyle)
        ax_mean.fill_between(x, MeanValues - CriticalValue * StdErrorValues,
                             MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal, color=color)
    ax_mean.set_xlabel("Percent of Total Data Labeled for Training")
    ax_mean.set_ylabel(Y_Label)
    ax_mean.set_title(Subtitle, fontsize=9)
    # ax_mean.legend(loc='upper right')
    ax_mean.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    if isinstance(xlim, list):
        ax_mean.set_xlim(xlim)

    ### Variance Plot ###
    fig_var = None
    if VarInput:
        fig_var, ax_var = plt.subplots(figsize=FigSize)
        for Label, VarianceValues in VarianceVector.items():
            num_iterations = len(VarianceValues)
            if num_iterations > 1:
                iterations_array = np.arange(num_iterations)
                x = (initial_train_proportion + (iterations_array / (num_iterations - 1)) * candidate_pool_proportion) * 100
            else:
                x = [initial_train_proportion * 100]
            color = Colors.get(Label, None) if Colors else None
            linestyle = Linestyles.get(Label, '-') if Linestyles else '-'
            legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
            ax_var.plot(x, VarianceValues, label=legend_label, color=color, linestyle=linestyle)
            lower_bound = StdErrorVarianceVector[Label]["lower"]
            upper_bound = StdErrorVarianceVector[Label]["upper"]
            ax_var.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)
        ax_var.set_xlabel("Percent of Total Data Labeled for Training")
        ax_var.set_ylabel("Variance of " + (Y_Label if Y_Label else "Error"))
        ax_var.set_title(Subtitle, fontsize=9)
        ax_var.legend(loc='upper right')
        if isinstance(xlim, list):
            ax_var.set_xlim(xlim)
    
    return (fig_mean, fig_var)

### Main Wrapper Function ###
### Main Wrapper Function ###
def generate_all_plots(aggregated_results_dir, image_dir):
    """
    Wrapper function to load aggregated .pkl files and generates all specified plots
    for both 'standard' and 'paper' evaluation metrics.
    """
    print("--- Starting Plot Generation from Aggregated Results ---")
    
    ### Aesthetics and Plot Definitions ###
    master_colors = {
        'Passive Learning': 'gray', 'GSx': 'cornflowerblue', 'GSy': 'salmon', 'iGS': 'red',
        'WiGS (Static w_x=0.75)': 'lightgreen', 'WiGS (Static w_x=0.5)': 'forestgreen',
        'WiGS (Static w_x=0.25)': 'darkgreen', 'WiGS (Time-Decay, Linear)': 'orange',
        'WiGS (Time-Decay, Exponential)': 'saddlebrown', 'WiGS (MAB-UCB1, c=0.5)': 'orchid',
        'WiGS (MAB-UCB1, c=2.0)': 'darkviolet', 'WiGS (MAB-UCB1, c=5.0)': 'indigo',
        'iRDM': 'darkcyan',
        'IDEAL': 'deeppink'
    }
    master_linestyles = {
        'Passive Learning': ':', 'GSx': ':', 'GSy': ':', 'iGS': '-',
        'WiGS (Static w_x=0.75)': '-', 'WiGS (Static w_x=0.5)': '-.',
        'WiGS (Static w_x=0.25)': '--', 'WiGS (Time-Decay, Linear)': '-',
        'WiGS (Time-Decay, Exponential)': '-.', 'WiGS (MAB-UCB1, c=0.5)': '-',
        'WiGS (MAB-UCB1, c=2.0)': '-', 'WiGS (MAB-UCB1, c=5.0)': '-',
        'iRDM': '--',
        'IDEAL': '-.'
    }
    master_legend = {
        'Passive Learning': 'Random', 'GSx': 'GSx', 'GSy': 'GSy', 'iGS': 'iGS',
        'WiGS (Static w_x=0.75)': 'WiGS (Static, w_x=0.75)', 'WiGS (Static w_x=0.5)': 'WiGS (Static, w_x=0.5)',
        'WiGS (Static w_x=0.25)': 'WiGS (Static, w_x=0.25)', 'WiGS (Time-Decay, Linear)': 'WiGS (Linear Decay)',
        'WiGS (Time-Decay, Exponential)': 'WiGS (Exponential Decay)', 'WiGS (MAB-UCB1, c=0.5)': 'WiGS (MAB, c=0.5)',
        'WiGS (MAB-UCB1, c=2.0)': 'WiGS (MAB, c=2.0)', 'WiGS (MAB-UCB1, c=5.0)': 'WiGS (MAB, c=5.0)',
        'iRDM': 'iRDM',
        'IDEAL': 'IDEAL'
    }
    
    ### Set up ###
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CC']
    plot_types = {'trace': None, 'trace_relative_iGS': 'iGS'}
    eval_types = ['standard', 'paper']

    ### Dynamically find datasets ###
    dataset_folders = [d for d in os.listdir(aggregated_results_dir) if os.path.isdir(os.path.join(aggregated_results_dir, d))]

    for data_name in dataset_folders:
        print(f"\nProcessing dataset: {data_name}...")
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
                                                                        initial_train_proportion=0.16, 
                                                                        candidate_pool_proportion=0.64,
                                                                        **results_for_metric)
                    
                    # Update the base saving path to include the evaluation type
                    base_plot_path = os.path.join(image_dir, eval_type, metric, folder_name)
                    os.makedirs(os.path.join(base_plot_path, 'trace'), exist_ok=True)
                    os.makedirs(os.path.join(base_plot_path, 'variance'), exist_ok=True)

                    # Save the mean trace plot
                    trace_plot_path = os.path.join(base_plot_path, 'trace', f"{data_name}_{metric}_TracePlot.png")
                    TracePlotMean.savefig(trace_plot_path, bbox_inches='tight', dpi=300)
                    plt.close(TracePlotMean)

                    # Save the variance trace plot
                    if TracePlotVariance:
                        variance_plot_path = os.path.join(base_plot_path, 'variance', f"{data_name}_{metric}_VariancePlot.png")
                        TracePlotVariance.savefig(variance_plot_path, bbox_inches='tight', dpi=300)
                        plt.close(TracePlotVariance)

        print(f"Finished all plots for {data_name}.")
    print("\n--- Plot Generation Complete ---")