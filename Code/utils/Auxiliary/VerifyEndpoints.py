### LIBRARIES ###
import pandas as pd
import numpy as np
import os
import pickle
import argparse
import sys

### Point check ###
def check_endpoints(dataset, aggregated_dir):
    """
    Loads all aggregated metrics for a dataset and checks if the
    first and last iteration values are identical across all strategies.
    """
    
    metrics_to_check = ['RMSE', 'MAE', 'R2', 'CC']
    base_path = os.path.join(aggregated_dir, dataset, "full_pool_metrics")
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found: {base_path}", file=sys.stderr)
        return

    print(f"--- Verification for Dataset: {dataset} ---")

    for metric in metrics_to_check:
        pkl_path = os.path.join(base_path, f"{metric}.pkl")
        if not os.path.exists(pkl_path):
            print(f"--- SKIPPING Metric: {metric} (File not found) ---")
            continue

        print(f"--- Checking Metric: {metric} ---")
        
        with open(pkl_path, 'rb') as f:
            results_dict = pickle.load(f)

        endpoint_data = []
        
        for strategy, df in results_dict.items():
            if df.empty:
                continue
            
            first_values = df.iloc[0]
            last_values = df.iloc[-1]
            
            mean_start = first_values.mean()
            mean_end = last_values.mean()
            
            endpoint_data.append({
                "Strategy": strategy,
                "Start_Value (Avg)": mean_start,
                "End_Value (Avg)": mean_end
            })

        if not endpoint_data:
            print("  No data found for this metric.")
            continue

        results_df = pd.DataFrame(endpoint_data).set_index("Strategy")
        
        # Check for consistency
        start_vals = results_df['Start_Value (Avg)']
        end_vals = results_df['End_Value (Avg)']
        is_start_consistent = np.all(np.isclose(start_vals, start_vals.iloc[0]))
        
        # Check if all values in the 'End_Value (Avg)' column are identical
        is_end_consistent = np.all(np.isclose(end_vals, end_vals.iloc[0]))

        if is_start_consistent:
            print(f"  [SUCCESS] All Start_Values are identical.")
        else:
            print(f"  [!!WARNING!!] Start_Values are NOT identical.")
            print(start_vals)
            
        if is_end_consistent:
            print(f"  [SUCCESS] All End_Values are identical.")
        else:
            print(f"  [!!WARNING!!] End_Values are NOT identical.")
            print(end_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check start/end points of aggregated results.")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset folder to analyze (e.g., "dgp_two_regime")')
    args = parser.parse_args()

    # --- Define Project Root and CD ---
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        # Fallback if running from root
        PROJECT_ROOT = os.getcwd() 
    
    # Change to project root so relative paths (e.g., "Results/...") work
    # This check is good practice
    if os.path.basename(PROJECT_ROOT) != 'WeightedGreedySampling':
        # If the script is run from inside Code/utils/Auxiliary, go up
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
        if os.path.basename(PROJECT_ROOT) != 'WeightedGreedySampling':
             print("Error: Could not find project root. Run this script from the project root directory.")
             sys.exit(1)

    os.chdir(PROJECT_ROOT)
    
    AGG_DIR = "Results/simulation_results/aggregated"
    
    check_endpoints(args.dataset, AGG_DIR)

# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "beer"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "burbidge_low_noise"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "concrete_cs"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "cps_wage"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "housing"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "pm10"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "wine_white"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "bodyfat"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "burbidge_misspecified"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "concrete_flow"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "dgp_three_regime"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "mpg"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "qsar"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "yacht"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "burbidge_correct"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "concrete_4"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "concrete_slump"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "dgp_two_regime"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "no2"
# python ./Code/utils/Auxiliary/VerifyEndpoints.py --dataset "wine_red"