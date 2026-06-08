#!/bin/bash
#SBATCH --job-name=AL_beer
#SBATCH --partition=short
#SBATCH --array=1-25
#SBATCH --output=/home/cregan/documents/code/00_LORIA/IAlefeu/WiGS_hydra_lightning/Code/Cluster/RunSimulations/ClusterMessages/out/beer_%A_%a.out
#SBATCH --error=/home/cregan/documents/code/00_LORIA/IAlefeu/WiGS_hydra_lightning/Code/Cluster/RunSimulations/ClusterMessages/error/beer_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=3000MB
#SBATCH --cpus-per-task=1

cd /home/cregan/documents/code/00_LORIA/IAlefeu/WiGS_hydra_lightning/Code
export PYTHONDONTWRITEBYTECODE=1

python RunSimulation.py \
    --Data "beer" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 25 \
    --CandidateProportion 0.95
