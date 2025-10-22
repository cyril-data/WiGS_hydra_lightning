### Packages ###
import os
import sys

### Paths ###
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)

### Create master sbatch function ###
from utils.Auxiliary.GenerateJobs import create_master_sbatch

### Execute ###
if __name__ == "__main__":
    
    ## Directories ##
    PROJECT_ROOT = os.path.dirname(CODE_DIR)
    DATA_DIRECTORY = os.path.join(PROJECT_ROOT, 'Data', 'processed')

    ## Cluster Parameters ##
    partition_name_input = "short"
    time_limit_input='11:59:59'
    memory_input='750MB'
    
    ## Define Simulation Parameters ##
    N_REPLICATIONS = 3
    CANDIDATE_PROPORTION = 0.8
    models_to_run = [
        'RidgeRegressionPredictor'
    ]
    
    ## Data sets ##
    pkl_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.pkl')]
    datasets_to_run = sorted([os.path.splitext(f)[0] for f in pkl_files])
    
    print("--- Starting sbatch file generation ---")
    
    ## Loop through datasets ##
    for dataset_name in datasets_to_run:
        print(f"Generating job file for dataset: {dataset_name}...")
        
        # Create path #
        model_sbatch_path = os.path.join(CODE_DIR, 'Cluster', 'RunSimulations', f"master_job_{dataset_name}.sbatch")
        
        # Create master sbatch #
        create_master_sbatch(
            partition_name = partition_name_input,
            time_limit=time_limit_input,
            memory=memory_input,
            sbatch_path=model_sbatch_path,
            n_replications=N_REPLICATIONS,
            n_models=len(models_to_run), 
            dataset_name=dataset_name,   
            candidate_prop=CANDIDATE_PROPORTION,
            code_dir=CODE_DIR
        )
    print("--- Finished ---")