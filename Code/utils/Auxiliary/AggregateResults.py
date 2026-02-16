### Import Packages ###
import os
import pickle
import glob
import pandas as pd

### Function ###
def AggregateResults(raw_results_dir, aggregated_results_dir):
    """
    Aggregates results from the simulation raw .pkl files, splitting results
    by evaluation type ('Standard' and 'Paper').
    """

    ### Set up ###
    print("--- Starting Aggregation of Raw Results ---")

    ### Discover Datasets ###
    search_pattern_all = os.path.join(raw_results_dir, '**', '*.pkl')
    all_pkl_files = glob.glob(search_pattern_all, recursive=True)
    if not all_pkl_files:
        print("No raw result files found. Exiting.")
        return
    dataset_basenames = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in all_pkl_files])))

    ### Loop through datasets ###
    for data_name in dataset_basenames:
        print(f"\nAggregating dataset: {data_name}...")
        search_pattern_dataset = os.path.join(raw_results_dir, data_name, f"{data_name}_*_seed_*.pkl")
        result_files_for_dataset = glob.glob(search_pattern_dataset)

        ## Load one file to inspect the structure ##
        with open(result_files_for_dataset[0], 'rb') as f:
            first_result = pickle.load(f)
        
        ## Discover metrics and eval types from DataFrame structure ##
        strategies = list(first_result.keys())
        error_df_template = first_result[strategies[0]]['ErrorVecs']
        eval_types = list(error_df_template.columns)  
        metrics = list(error_df_template.index)      
                
        aggregated_data = {
            s: {
                'ErrorVecs': {
                    eval_type: {m: [] for m in metrics} for eval_type in eval_types
                },
                'ElapsedTime': [],
                'SelectionHistory': [],
                'WeightHistory': []} for s in strategies
        }

        ## Aggregation Loop ##
        all_initial_indices_list = []
        for i, file_path in enumerate(result_files_for_dataset):
            with open(file_path, 'rb') as f:
                single_run_result = pickle.load(f)

            # Grab initial indices from the first strategy (they are identical for this seed)
            first_strategy_key = list(single_run_result.keys())[0]
            if 'InitialTrainIndices' in single_run_result[first_strategy_key]:
                all_initial_indices_list.append(single_run_result[first_strategy_key]['InitialTrainIndices'])
            
            for strategy, results in single_run_result.items():
                if strategy in aggregated_data:
                    
                    error_vec_df = results['ErrorVecs']
                    for eval_type in error_vec_df.columns:
                        for metric in error_vec_df.index:
                            metric_values_list = error_vec_df.loc[metric, eval_type]
                            series = pd.Series(metric_values_list, name=f"Sim_{i}")
                            aggregated_data[strategy]['ErrorVecs'][eval_type][metric].append(series)

                    # Store #
                    aggregated_data[strategy]['ElapsedTime'].append(results['ElapsedTime'])
                    aggregated_data[strategy]['SelectionHistory'].append(results['SelectionHistory'])
                    if 'WeightHistory' in results:
                        aggregated_data[strategy]['WeightHistory'].append(results['WeightHistory'])

        ## Final Processing and Saving ##
        dataset_output_dir = os.path.join(aggregated_results_dir, data_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        for eval_type in eval_types:
            eval_metrics_save_dir = os.path.join(dataset_output_dir, f'{eval_type.lower()}_metrics')
            os.makedirs(eval_metrics_save_dir, exist_ok=True)
            
            for metric in metrics:
                metric_results = {}
                for strategy in strategies:
                    series_list = aggregated_data[strategy]['ErrorVecs'][eval_type][metric]
                    if series_list:
                        metric_results[strategy] = pd.concat(series_list, axis=1)
                
                if metric_results:
                    output_path = os.path.join(eval_metrics_save_dir, f"{metric}.pkl")
                    with open(output_path, 'wb') as f:
                        pickle.dump(metric_results, f)
                    print(f"  > Saved {metric}.pkl to {eval_type.lower()}_metrics/")

        time_data = {strategy: data['ElapsedTime'] for strategy, data in aggregated_data.items()}
        time_df = pd.DataFrame(time_data)
        time_df.to_csv(os.path.join(dataset_output_dir, 'ElapsedTime.csv'), index_label='Simulation')
        print(f"  > Saved ElapsedTime.csv")
        
        ### Save Selection History ###
        history_save_dir = os.path.join(dataset_output_dir, 'selection_history')
        os.makedirs(history_save_dir, exist_ok=True)
        for strategy in strategies:
            history_data = aggregated_data[strategy]['SelectionHistory']
            history_df = pd.DataFrame(history_data).transpose()
            history_df.columns = [f"Sim_{i}" for i in range(len(history_data))]
            history_df.to_csv(os.path.join(history_save_dir, f'{strategy}_SelectionHistory.csv'), index_label='Iteration')
        print(f"  > Saved SelectionHistory CSVs.")

        ### Save Weight History ###
        weight_save_dir = os.path.join(dataset_output_dir, 'weight_history')
        os.makedirs(weight_save_dir, exist_ok=True)
        files_saved_count = 0 
        for strategy in strategies:
            weight_data = aggregated_data[strategy]['WeightHistory']
            if weight_data: 
                weight_df = pd.DataFrame(weight_data).transpose()                
                if weight_df.notna().any().any():
                    weight_df.columns = [f"Sim_{i}" for i in range(len(weight_data))]
                    weight_df.to_csv(os.path.join(weight_save_dir, f'{strategy}_WeightHistory.csv'), index_label='Iteration')
                    files_saved_count += 1         
        if files_saved_count > 0:
            print(f"  > Saved WeightHistory CSVs.")
        else:
            print(f"  > No WeightHistory CSVs needed.")

        # Save InitialIndices file per dataset #
        if all_initial_indices_list:
            try:
                indices_df = pd.DataFrame(all_initial_indices_list).transpose()
                indices_df.columns = [f"Sim_{i}" for i in range(len(all_initial_indices_list))]
                indices_output_path = os.path.join(dataset_output_dir, 'InitialIndices.csv')
                indices_df.to_csv(indices_output_path, index_label='Index_Position')
                print(f"  > Saved InitialIndices.csv")
            except ValueError as e:
                print(f"  > Error creating/saving InitialIndices.csv for {data_name}: {e}")
        else:
             print(f"  > Warning: InitialIndices data not found in raw files for {data_name}.")
    print("\n--- Aggregation Complete ---")

### MAIN ###
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'raw')
    AGGREGATED_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results', 'simulation_results', 'aggregated')
    
    AggregateResults(raw_results_dir=RAW_RESULTS_DIR, aggregated_results_dir=AGGREGATED_RESULTS_DIR)