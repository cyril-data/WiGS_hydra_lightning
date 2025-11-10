#!/bin/bash

#SBATCH --job-name=GeneratePlots
#SBATCH --partition=short
#SBATCH --array=1-20
#SBATCH --output=RunSimulations/ClusterMessages/out/genplots_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/genplots_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4GB

### --- Script Execution --- ###
echo "--- Starting plot generation for Task ${SLURM_ARRAY_TASK_ID} ---"

cd "$SLURM_SUBMIT_DIR/../.."
echo "Current Directory: $(pwd)"

AGG_DIR="Results/simulation_results/aggregated"
DATASETS=($(find "$AGG_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))

task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_TO_PROCESS=${DATASETS[$task_id_zero_based]}

if [ -z "$DATASET_TO_PROCESS" ]; then
    echo "Error: Task ID ${SLURM_ARRAY_TASK_ID} is out of bounds. No dataset found."
    exit 1
fi

echo "Processing dataset: ${DATASET_TO_PROCESS}"
# python Code/utils/Auxiliary/GeneratePlots.py --dataset "$DATASET_TO_PROCESS"
python Code/utils/Auxiliary/GeneratePlots.py --dataset "$DATASET_TO_PROCESS" --no-legend

echo "--- Finished Task ${SLURM_ARRAY_TASK_ID} ---"