#!/bin/bash
#SBATCH --job-name=yacht_LinearRegressionPredictor
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --output=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/yacht_LinearRegressionPredictor_%j.out
#SBATCH --error=/mnt/beegfs/homes/simondn/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/yacht_LinearRegressionPredictor_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu

# --- Set the correct working directory ---
cd /mnt/beegfs/homes/simondn/WeightedGreedySampling/Code

# Command to execute your main Python simulation script
# Now we can just use the script's filename
python RunSimulation.py \
    --Data "yacht" \
    --NSim 100 \
    --ModelType "LinearRegressionPredictor" \
    --TestProportion 0.2 \
    --CandidateProportion 0.8 \
    --Output "yacht_results.pkl"
