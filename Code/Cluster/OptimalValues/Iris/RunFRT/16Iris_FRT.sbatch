#!/bin/bash
#SBATCH --job-name=16Iris_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_16Iris_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_16Iris_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 16Iris_FRT \
    --Data Iris \
    --Seed 16 \
    --TestProportion 0.25 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.05 \
    --Output 16Iris_FRT.pkl