#!/bin/bash

# Récupérer toute la commande
cmd="$*"

if [ -z "$cmd" ]; then
    echo "Erreur : aucune commande fournie."
    echo "Usage : $0 \"python RunSimulation.py --Data hydralightning --TaskID 0 ...\""
    exit 1
fi

# Extraire tout ce qui suit "--k_top"
base_name=$(echo "$cmd" | sed -n 's/.*--k_top //p')

# Supprimer les espaces
base_name=$(echo "$base_name" | tr -d ' ')

# Nettoyage : suppression des caractères non-safe pour un nom de fichier/job
base_name=$(echo "$base_name" | tr -cd '[:alnum:]_.-')

# Générer le nom du job
job_name="${base_name}_$(date +%Y%m%d_%H%M%S)"

echo "job_name ${job_name}"

# Génère le fichier .slurm

#!/bin/bash
CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda

echo $CONDA_PATH_INSTALL

cat > "temp_${job_name}.slurm" <<EOL_
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.err
#SBATCH --error=logs/${job_name}.err
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --account hir@h100
#SBATCH -C h100
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

${cmd}

EOL_

echo "${cmd}"

# Submit job
sbatch "temp_${job_name}.slurm"

# Optionnal : Remove temps slurm file after submit
rm "temp_${job_name}.slurm"

