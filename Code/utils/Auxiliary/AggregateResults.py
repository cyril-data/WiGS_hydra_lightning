### Import Packages ###
import os
import pickle
import glob
import pandas as pd

### Function ###
def AggregateResults(raw_results_dir, aggregated_results_dir):
    """
    Aggregates results from the simulation raw .pkl files.

    Args:
        raw_results_dir: Directory location of the raw .pkl files.
        aggregated_results_dir: Target irectory location of the aggregated results.
    """

    ### Set up ###
    print("--- Starting Aggregation of Raw Results ---")

    ### Discover Datasets ###
    search_pattern_all = os.path.join(raw_results_dir, '**', '*.pkl')
    all_pkl_files = glob.glob(search_pattern_all, recursive=True)
    dataset_basenames = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in all_pkl_files])))

    ### Loop through datasets ###
    for data_name in dataset_basenames:
        print(f"\nAggregating dataset: {data_name}...")
        search_pattern_dataset = os.path.join(raw_results_dir, data_name, f"{data_name}_*_seed_*.pkl")
        result_files_for_dataset = glob.glob(search_pattern_dataset)

        ## Load data ##
        with open(result_files_for_dataset[0], 'rb') as f:
            first_result = pickle.load(f)
        
        ## Initialize results storage places ##
        strategies = list(first_result.keys())
        metrics = list(first_result[strategies[0]]['ErrorVec'].columns)
        aggregated_data = {s: {'ErrorVec': {m: [] for m in metrics}, 'ElapsedTime': [], 'SelectionHistory': []} for s in strategies}

        ## Aggregation Loop ##
        for i, file_path in enumerate(result_files_for_dataset):
            with open(file_path, 'rb') as f:
                single_run_result = pickle.load(f)
            
            for strategy, results in single_run_result.items():
                if strategy in aggregated_data:

                    # Collect Error Metrics
                    for metric, series in results['ErrorVec'].items():
                        series.name = f"Sim_{i}"
                        aggregated_data[strategy]['ErrorVec'][metric].append(series)
                    
                    # Collect Elapsed Time
                    aggregated_data[strategy]['ElapsedTime'].append(results['ElapsedTime'])
                    
                    # Collect Selection History
                    aggregated_data[strategy]['SelectionHistory'].append(results['SelectionHistory'])

        ## Final Processing and Saving ##
        dataset_output_dir = os.path.join(aggregated_results_dir, data_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        metrics_save_dir = os.path.join(dataset_output_dir, 'metrics')
        history_save_dir = os.path.join(dataset_output_dir, 'selection_history')
        os.makedirs(metrics_save_dir, exist_ok=True)
        os.makedirs(history_save_dir, exist_ok=True)
        
        ## Save Aggregated Metrics ##
        for metric in metrics:
            metric_results = {}
            for strategy in strategies:
                series_list = aggregated_data[strategy]['ErrorVec'][metric]
                if series_list:
                    metric_results[strategy] = pd.concat(series_list, axis=1)
            
            if metric_results:
                output_path = os.path.join(metrics_save_dir, f"{metric}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(metric_results, f)
                print(f"  > Saved {metric}.pkl")

        ## Save Aggregated Elapsed Time ##
        time_data = {strategy: data['ElapsedTime'] for strategy, data in aggregated_data.items()}
        time_df = pd.DataFrame(time_data)
        time_df.to_csv(os.path.join(dataset_output_dir, 'ElapsedTime.csv'), index_label='Simulation')
        print(f"  > Saved ElapsedTime.csv")
        
        ## Save Aggregated Selection History ##
        for strategy in strategies:
            history_data = aggregated_data[strategy]['SelectionHistory']
            history_df = pd.DataFrame(history_data).transpose()
            history_df.columns = [f"Sim_{i}" for i in range(len(history_data))]
            history_df.to_csv(os.path.join(history_save_dir, f'{strategy}_SelectionHistory.csv'), index_label='Iteration')
        print(f"  > Saved SelectionHistory CSVs.")

    print("\n--- Aggregation Complete ---")

### Execute ###
if __name__ == "__main__":

    ## Directories ##
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'raw')
    AGGREGATED_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    
    ## Aggregate results ##
    AggregateResults(raw_results_dir=RAW_RESULTS_DIR, aggregated_results_dir=AGGREGATED_RESULTS_DIR)