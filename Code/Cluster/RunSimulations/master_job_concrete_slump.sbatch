#!/bin/bash
#SBATCH --job-name=AL_concrete_slump
#SBATCH --partition=short
#SBATCH --array=1-3
#SBATCH --output=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/concrete_slump_%A_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/concrete_slump_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1

cd /mnt/beegfs/homes/simondn/WeightedGreedySampling/Code

python RunSimulation.py \
    --Data "concrete_slump" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 3 \
    --CandidateProportion 0.95
