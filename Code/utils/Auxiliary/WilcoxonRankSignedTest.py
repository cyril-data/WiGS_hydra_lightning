### Packages ###
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import Dict, Optional
import os
import pickle
import argparse
import sys

### Function ###
### Function ###
def WilcoxonRankSignedTest(SimulationErrorResults: Dict[str, pd.DataFrame],
                           RoundingVal: Optional[int] = 3) -> pd.DataFrame:
    """
    Performs a pairwise Wilcoxon signed-rank test on simulation results.
    """

    ### Set Up ###
    strategies = list(SimulationErrorResults.keys())
    n_strategies = len(strategies)
    PValeMatrix = np.zeros((n_strategies, n_strategies)) * np.nan 

    ### Wilcoxon Signed-Rank Test ###
    for i in range(n_strategies):
        for j in range(i):
            mean_error_i = np.mean(SimulationErrorResults[strategies[i]], axis=0)
            mean_error_j = np.mean(SimulationErrorResults[strategies[j]], axis=0)
            
            if np.allclose(mean_error_i, mean_error_j):
                pval = 1.0
            else:
                stat, pval = wilcoxon(mean_error_i, mean_error_j, zero_method='zsplit')
            
            PValeMatrix[i, j] = pval

    ### Formatting ###
    np.fill_diagonal(PValeMatrix, 1.0)
    pval_df = pd.DataFrame(PValeMatrix, index=strategies, columns=strategies)
    mask = np.tril(np.ones(pval_df.shape), k=0).astype(bool)
    
    def format_pvalue(p):
        if pd.isna(p): 
            return ""  
        if p == 1.0: 
            return "$1.000$" 
        if p < 0.001:
            return "$<0.001$" 
        return f"${p:.{RoundingVal}f}$" 
    WRSTResults = pval_df.where(mask).map(format_pvalue)

    ### Return ###
    return WRSTResults

# -----------------------------------------------------------------
# --- MAIN SCRIPT (NOW INCLUDES NAME MAPPING) ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Wilcoxon test on aggregated simulation results.")
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset folder to analyze (e.g., "dgp_three_regime")')
    parser.add_argument('--metric', type=str, default='RMSE',
                        help="Metric to test (e.g., 'RMSE', 'MAE'). Default is 'RMSE'.")
    parser.add_argument('--eval_type', type=str, default='full_pool',
                        help="Evaluation type (e.g., 'full_pool'). Default is 'full_pool'.")

    args = parser.parse_args()

    ### --- Define Paths --- ###
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    AGGREGATED_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    OUTPUT_TABLE_DIR = os.path.join(PROJECT_ROOT, 'Results', 'tables')
    os.makedirs(OUTPUT_TABLE_DIR, exist_ok=True) 

    metric_pkl_path = os.path.join(AGGREGATED_RESULTS_DIR, 
                                   args.dataset, 
                                   f"{args.eval_type}_metrics", 
                                   f"{args.metric}.pkl")

    ### --- Load Data --- ###
    if not os.path.exists(metric_pkl_path):
        print(f"Error: File not found at path: {metric_pkl_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading data from: {metric_pkl_path}\n")
    with open(metric_pkl_path, 'rb') as f:
        simulation_results = pickle.load(f)

    ### --- UPDATED: Apply Name Mapping --- ###
    # This dictionary maps the long, raw names to short, publication-ready names
    NAME_MAPPING = {
        'Passive Learning': 'Passive',
        'GSx': 'GSx',
        'GSy': 'GSy',
        'iGS': 'iGS',
        'WiGS (Static w_x=0.25)': 'WiGS-S (0.25)',
        'WiGS (Static w_x=0.5)': 'WiGS-S (0.50)',
        'WiGS (Static w_x=0.75)': 'WiGS-S (0.75)',
        'WiGS (Time-Decay, Linear)': 'WiGS-Lin',
        'WiGS (Time-Decay, Exponential)': 'WiGS-Exp',
        'WiGS (MAB-UCB1, c=0.5)': 'WiGS-MAB (c=0.5)',
        'WiGS (MAB-UCB1, c=2.0)': 'WiGS-MAB (c=2.0)',
        'WiGS (MAB-UCB1, c=5.0)': 'WiGS-MAB (c=5.0)',
        'WiGS (SAC)': 'WiGS-SAC'
    }

    # Create a new dictionary, applying the short names
    filtered_results = {}
    for long_name, data in simulation_results.items():
        # Use the short name if it exists, otherwise keep the long name
        short_name = NAME_MAPPING.get(long_name, long_name)
        filtered_results[short_name] = data

    print(f"--- Wilcoxon Signed-Rank Test Results for: {args.dataset} (Metric: {args.metric}) ---")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Pass the dictionary with short names to the function
    results_table = WilcoxonRankSignedTest(filtered_results, RoundingVal=3)
    print(results_table)

    # --- Save to LaTeX ---
    latex_filename = f"{args.dataset}_{args.metric}_{args.eval_type}_wilcoxon.tex"
    latex_full_path = os.path.join(OUTPUT_TABLE_DIR, latex_filename)
    
    # escape=False is CRITICAL to allow the "$" and "<" characters
    results_table.to_latex(latex_full_path, escape=False)
    
    print(f"\n--- LaTeX table saved to: {latex_full_path} ---")

    print("\n--- Notes ---")
    print("The table shows p-values for the null hypothesis that the two strategies are equivalent.")
    print("A small p-value (e.g., < 0.05) suggests a statistically significant difference.")
    print("Each test compares the paired vectors of average RMSE (one avg RMSE per simulation seed).")