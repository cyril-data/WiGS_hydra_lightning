### Import Packages ###
import os
import pickle
import glob
import pandas as pd

def AggregateResults(raw_results_dir, aggregated_results_dir):

    ### Set up ###
    print("--- Starting Aggregation of Raw Results ---")
    final_aggregated_data = {}
    
    ### Datasets ###
    try:
        search_pattern_all = os.path.join(raw_results_dir, '**', '*.pkl')
        all_pkl_files = glob.glob(search_pattern_all, recursive=True)
        dataset_basenames = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in all_pkl_files])))
        if not dataset_basenames:
            print(f"Warning: No datasets found in subdirectories of '{raw_results_dir}'.")
            print("--- Aggregation Complete ---")
            return
    except FileNotFoundError:
        print(f"Error: Raw results directory not found at '{raw_results_dir}'")
        return

    ### Loop through datasets ###
    for data_name in dataset_basenames:
        print(f"\nAggregating dataset: {data_name}...")
        search_pattern_dataset = os.path.join(raw_results_dir, data_name, f"{data_name}_*_seed_*.pkl")
        result_files_for_dataset = glob.glob(search_pattern_dataset)
        
        if not result_files_for_dataset:
            print(f"  > Warning: No result files found for {data_name}. Skipping.")
            continue

        ## Aggregation ##
        with open(result_files_for_dataset[0], 'rb') as f:
            first_result = pickle.load(f) 
        aggregated_results = {s: {m: [] for m in d.keys()} for s, d in first_result.items()}

        for i, file_path in enumerate(result_files_for_dataset):
            with open(file_path, 'rb') as f:
                single_run_result = pickle.load(f)
            for strategy, metrics_df in single_run_result.items():
                if strategy in aggregated_results:
                    for metric in metrics_df.columns:
                        if metric in aggregated_results[strategy]:
                            series = metrics_df[metric].copy()
                            series.name = f"Sim_{i}"
                            aggregated_results[strategy][metric].append(series)
        
        for strategy, metrics_dict in aggregated_results.items():
            final_aggregated_data[strategy] = {}
            for metric, series_list in metrics_dict.items():
                if series_list:
                    final_aggregated_data[strategy][metric] = pd.concat(series_list, axis=1)

        ## Save ##
        dataset_output_dir = os.path.join(aggregated_results_dir, data_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for metric in ['RMSE', 'MAE', 'R2', 'CC']:
            metric_results = {
                strategy: data[metric] 
                for strategy, data in final_aggregated_data.items() 
                if metric in data
            }
            
            if metric_results:
                output_path = os.path.join(dataset_output_dir, f"{metric}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(metric_results, f)
                print(f"  > Saved {metric}.pkl")

    print("\n--- Aggregation Complete ---")

### Execute ###
if __name__ == "__main__":

    ## Directories ##
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'raw')
    AGGREGATED_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    
    ## Execute ##
    AggregateResults(raw_results_dir=RAW_RESULTS_DIR, aggregated_results_dir=AGGREGATED_RESULTS_DIR)