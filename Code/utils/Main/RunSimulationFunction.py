### Import standard libraries ###
import pandas as pd
from tqdm import tqdm

### Import custom functions ###
from utils.Main.OneIterationFunction import OneIterationFunction

### Function Definition ###
def RunSimulationFunction(DataFileInput,
                          NSim,
                          machine_learning_model,
                          test_proportion,
                          candidate_proportion):

    ### Set Up ###
    all_results_by_strategy = {}
    strategies_to_run = {
        'Passive Learning': {'SelectorType': 'PassiveLearningSelector'},
        'GSx': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSx'},
        'GSy': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSy'},
        'iGS': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'iGS'},
        'WiGS (Static w_x=0.25)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                   'weight_strategy': 'static',
                                   'w_x': 0.25},
        'WiGS (Static w_x=0.5)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                  'weight_strategy': 'static',
                                  'w_x': 0.5},
        'WiGS (Static w_x=0.75)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                   'weight_strategy': 'static',
                                   'w_x': 0.75},
        'WiGS (Time-Decay, Linear)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                      'weight_strategy': 'time_decay',
                                      'decay_type': 'linear'},
        'WiGS (Time-Decay, Exponential)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                           'weight_strategy': 'time_decay',
                                           'decay_type': 'exponential',
                                           'decay_constant': 5.0},
        'WiGS (MAB-UCB1, c=0.5)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 0.5},
        'WiGS (MAB-UCB1, c=2.0)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 2.0},
        'WiGS (MAB-UCB1, c=5.0)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 5.0}
    }
    
    ### Main Simulation Loop ###
    for strategy_name, strategy_params in strategies_to_run.items():

        ## Set up ##
        print(f"\n--- Running Simulations for: {strategy_name} ---")
        metrics_collector = {'RMSE': [], 'MAE': [], 'R2': [], 'CC': []}
        
        ## Run one iteration NSim times ##
        for i in tqdm(range(NSim), desc="Simulations"):
            
            # Base configuration #
            SimulationConfigInput = {
                'DataFileInput': DataFileInput, 'Seed': i,
                'TestProportion': test_proportion, 'CandidateProportion': candidate_proportion,
                'ModelType': machine_learning_model
            }
            SimulationConfigInput.update(strategy_params)
            
            # Run the simulation #
            results = OneIterationFunction(SimulationConfigInput)

            # Extract Results #
            results_df = results["ErrorVec"]
            for metric in metrics_collector.keys():
                metric_series = results_df[metric].copy()
                metric_series.name = f'Sim_{i}'
                metrics_collector[metric].append(metric_series)
            
        # Store results #
        strategy_metric_dfs = {}
        for metric, series_list in metrics_collector.items():
            strategy_metric_dfs[metric] = pd.concat(series_list, axis=1)
        all_results_by_strategy[strategy_name] = strategy_metric_dfs

    ### Return the nested dictionary ###
    return all_results_by_strategy