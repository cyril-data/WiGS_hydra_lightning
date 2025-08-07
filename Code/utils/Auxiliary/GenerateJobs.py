### Packages ###
import os
import stat

### Functions ###
def create_sbatch_files(
    data_dir,
    sbatch_dir,
    results_dir,
    log_dir,
    python_script_path,
    n_sim,
    model_type,
    test_prop,
    candidate_prop,
    time_limit='01:00:00',
    memory='30000M'
):
    ### Create Output Directories ###
    os.makedirs(sbatch_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'out'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'error'), exist_ok=True)
    
    ### Get Data Files ###
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    ### Loop and Generate Scripts ###
    for data_file in data_files:

        ## Set up ##
        data_name = os.path.splitext(data_file)[0]
        job_name = f"{data_name}_{model_type}"
        output_filename = f"{data_name}_results.pkl"
        
        ## Define log file paths ##
        log_out_path = os.path.join(log_dir, 'out', f"{job_name}_%j.out")
        log_err_path = os.path.join(log_dir, 'error', f"{job_name}_%j.err")

        ## Define the content of the .sbatch file ##
        sbatch_content = f"""
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition=short
        #SBATCH --ntasks=1
        #SBATCH --output={log_out_path}
        #SBATCH --error={log_err_path}
        #SBATCH --time={time_limit}
        #SBATCH --mem-per-cpu={memory}
        #SBATCH --cpus-per-task=1
        #SBATCH --mail-type=ALL
        #SBATCH --mail-user=simondn@uw.edu

        cd {os.path.dirname(python_script_path)}
        python {os.path.basename(python_script_path)} \\
            --Data "{data_name}" \\
            --NSim {n_sim} \\
            --ModelType "{model_type}" \\
            --TestProportion {test_prop} \\
            --CandidateProportion {candidate_prop} \\
            --Output "{output_filename}"
        """

        ## Save ##
        sbatch_file_path = os.path.join(sbatch_dir, f"run_{data_name}_{model_type}.sbatch")
        with open(sbatch_file_path, 'w') as f:
            f.write(sbatch_content)
        os.chmod(sbatch_file_path, stat.S_IRWXU)

    print(f"Generated {len(data_files)} .sbatch files for model '{model_type}' in '{sbatch_dir}'")
