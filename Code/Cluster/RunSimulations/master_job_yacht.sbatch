#!/bin/bash
#SBATCH --job-name=AL_yacht
#SBATCH --partition=short
#SBATCH --array=1-100
#SBATCH --output=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/yacht_%A_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/yacht_%A_%a.err
#SBATCH --time=2:59:59
#SBATCH --mem-per-cpu=500MB
#SBATCH --cpus-per-task=1

cd /mnt/beegfs/homes/simondn/WeightedGreedySampling/Code

python RunSimulation.py \
    --Data "yacht" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --TestProportion 0.2 \
    --CandidateProportion 0.8
