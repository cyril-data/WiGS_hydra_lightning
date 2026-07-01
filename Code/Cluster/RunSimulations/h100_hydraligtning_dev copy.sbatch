#!/bin/bash
#SBATCH --job-name=wi_h100_2h
#SBATCH --output=logs/wi_h100_2h.err
#SBATCH --error=logs/wi_h100_2h.err
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --account hir@h100
#SBATCH -C        h100
#SBATCH --time=1:50:00
#SBATCH --qos=qos_gpu_h100-dev

CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda_wigs

module purge

# Load arch/h100
module load arch/h100

# Load miniforge
module load miniforge/24.11.3

# Load baseline env
conda activate pytorch-gpu-2.8.0+py3.12.11

# Activate env with --stack
conda activate --stack $CONDA_PATH_INSTALL

# Load dependancies
module load pytorch-gpu/py3/2.8.0

# Change PYTHONUSERBASE to be able to use bin in the env conda_path
export PYTHONUSERBASE=$CONDA_PATH_INSTALL

SLURM_ARRAY_TASK_ID=0

python RunSimulation.py --Data "hydralightning" --TaskID "$SLURM_ARRAY_TASK_ID" --NReplications 25 --CandidateProportion 0.95 --k_top 10000 --hl_xp baseline_active_learning_h100