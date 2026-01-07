#!/bin/bash
#SBATCH --job-name=AL_wine_white
#SBATCH --partition=short
#SBATCH --array=1-100
#SBATCH --output=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/wine_white_%A_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/wine_white_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --mem-per-cpu=1500MB
#SBATCH --cpus-per-task=1

cd /mnt/beegfs/homes/simondn/WeightedGreedySampling/Code

python RunSimulation.py \
    --Data "wine_white" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --CandidateProportion 0.95
