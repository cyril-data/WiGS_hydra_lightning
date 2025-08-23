import os
import stat

def create_master_sbatch(
    sbatch_path,
    n_replications,
    n_models,
    dataset_name,
    test_prop,
    candidate_prop,
    code_dir,
    time_limit='2:59:59',
    memory='500MB'
):
    total_jobs = n_models * n_replications
    log_dir = os.path.join(code_dir, 'Cluster', 'RunSimulations', 'ClusterMessages')
    python_script_name = 'RunSimulation.py'

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=AL_{dataset_name}
#SBATCH --partition=short
#SBATCH --array=1-{total_jobs}
#SBATCH --output={log_dir}/out/{dataset_name}_%A_%a.out
#SBATCH --error={log_dir}/error/{dataset_name}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={memory}
#SBATCH --cpus-per-task=1

cd {code_dir}

python {python_script_name} \\
    --Data "{dataset_name}" \\
    --TaskID "$SLURM_ARRAY_TASK_ID" \\
    --NReplications {n_replications} \\
    --TestProportion {test_prop} \\
    --CandidateProportion {candidate_prop}
"""
    
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU)
    print(f"  > Successfully generated master job script at '{os.path.basename(sbatch_path)}'")
    print(f"    Job array size: {total_jobs} ( {n_models} models x {n_replications} replications)")