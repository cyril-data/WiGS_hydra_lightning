#!/bin/bash
#SBATCH --job-name=AL_bodyfat
#SBATCH --partition=short
#SBATCH --array=1-100
#SBATCH --output=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/bodyfat_%A_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/bodyfat_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1

cd /mnt/beegfs/homes/simondn/WeightedGreedySampling/Code

python RunSimulation.py \
    --Data "bodyfat" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --CandidateProportion 0.8
