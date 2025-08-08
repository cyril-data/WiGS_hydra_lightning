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
    all_error_vectors = {}
    strategies_to_run = {
        # 'Passive Learning': {'SelectorType': 'PassiveLearningSelector'},
        # 'GSx': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSx'},
        # 'GSy': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSy'},
        'iGS': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'iGS'},
        'WiGS (Static w_x=0.25)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                   'weight_strategy': 'static',
                                   'w_x': 0.25},                                     # Exploitation-focused
        'WiGS (Static w_x=0.5)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                  'weight_strategy': 'static',
                                  'w_x': 0.5},
        'WiGS (Static w_x=0.75)': {'SelectorType': 'WeightedGreedySamplingSelector', # Balanced
                                   'weight_strategy': 'static',
                                   'w_x': 0.75},                                     # Exploration-focused
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
        print(f"\n--- Running Simulations for: {strategy_name} ---")
        current_error_vector = []
        
        ## Run one iteration NSim times ##
        for i in tqdm(range(NSim), desc="Simulations"):
            
            # Base configuration
            SimulationConfigInput = {
                'DataFileInput': DataFileInput,
                'Seed': i,
                'TestProportion': test_proportion,
                'CandidateProportion': candidate_proportion,
                'ModelType': machine_learning_model
            }
            
            # Add the specific strategy parameters
            SimulationConfigInput.update(strategy_params)
            
            # Run the simulation
            results = OneIterationFunction(SimulationConfigInput)
            
            # Extract and store errors
            results_df = results["ErrorVec"].copy()
            results_df.rename(columns={'Error': f'Sim_{i}_Error'}, inplace=True)
            current_error_vector.append(results_df)
            
        # Consolidate results for the current strategy
        all_error_vectors[strategy_name] = pd.concat(current_error_vector, axis=1)

    ### Return the collected results ###
    return all_error_vectors