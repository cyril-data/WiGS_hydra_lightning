### Packages ###
import os
import stat

### Function ###
def create_master_sbatch(partition_name, 
                         sbatch_path, 
                         n_replications, 
                         n_models, 
                         dataset_name, 
                         candidate_prop, 
                         code_dir, 
                         time_limit='2:59:59', 
                         memory='500MB'):
    """
    Generates a master SLURM sbatch script for running a job array.

    Args:
        partition_name (str): The name of the SLURM partition to submit the job to
        sbatch_path (str): The full file path where the generated .sbatch script will be saved.
        n_replications (int): The number of times each simulation is replicated with a different seed.
        n_models (int): The total number of different machine learning models being tested in this job array.
        dataset_name (str): The base name of the dataset this job array will process.
        candidate_prop (float): The proportion of the data to be used for the initial candidate pool.
        code_dir (str): The absolute path to the main 'Code/' directory of the project. 
        time_limit (str): The maximum wall time for each job in the array formatted as 'HH:MM:SS'.
        memory (str): The amount of memory to request for each job including units. Defaults to '500MB'.
    """

    ### Set up ##
    total_jobs = n_models * n_replications
    log_dir = os.path.join(code_dir, 'Cluster', 'RunSimulations', 'ClusterMessages')
    python_script_name = 'RunSimulation.py'

### sbatch content ###
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=AL_{dataset_name}
#SBATCH --partition={partition_name}
#SBATCH --array=1-{total_jobs}
#SBATCH --output={log_dir}/out/{dataset_name}_%A_%a.out
#SBATCH --error={log_dir}/error/{dataset_name}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={memory}
#SBATCH --cpus-per-task=1

cd {code_dir}
export PYTHONDONTWRITEBYTECODE=1

python {python_script_name} \\
    --Data "{dataset_name}" \\
    --TaskID "$SLURM_ARRAY_TASK_ID" \\
    --NReplications {n_replications} \\
    --CandidateProportion {candidate_prop}
"""
    
    ### Save sbatch content ###
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU)
    print(f"  > Successfully generated master job script at '{os.path.basename(sbatch_path)}'")
    print(f"    Job array size: {total_jobs} ( {n_models} models x {n_replications} replications)")