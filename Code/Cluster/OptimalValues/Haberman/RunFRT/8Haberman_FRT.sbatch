#!/bin/bash
#SBATCH --job-name=8Haberman_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_8Haberman_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_8Haberman_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 8Haberman_FRT \
    --Data Haberman \
    --Seed 8 \
    --TestProportion 0.25 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.2 \
    --Output 8Haberman_FRT.pkl