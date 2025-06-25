#!/bin/bash
#SBATCH --job-name=46BankNote_FRT
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH -o ClusterMessages/out/myscript_46BankNote_FRT_%j.out
#SBATCH -e ClusterMessages/error/myscript_46BankNote_FRT_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd ~/RashomonActiveLearning
module load Python
python Code/OptimalThresholdSimulation.py \
    --JobName 46BankNote_FRT \
    --Data BankNote \
    --Seed 46 \
    --TestProportion 0.25 \
    --regularization 0.01 \
    --RashomonThresholdType Adder \
    --RashomonThreshold 0.5 \
    --Output 46BankNote_FRT.pkl