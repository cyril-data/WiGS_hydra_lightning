#!/bin/bash

# ======================
# 1. Valeurs par défaut
# ======================
DATA='hydralightning'
TASKID=0
NREPLICATIONS=25
CANDIDATEPROPORTION=0.95
HL_XP='100k_active_learning_h100'
NO_CV=1
STRAT='iGS'
HL_MAX_EPOCH=50
K_TOP=1000
SUBSET_RAND_CANDIDAT=-1
CURRICULUM=-1

# ======================
# 2. Parsing des arguments
# ======================
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --DATA) DATA="$2"; shift ;;
        --TASKID) TASKID="$2"; shift ;;
        --NREPLICATIONS) NREPLICATIONS="$2"; shift ;;
        --CANDIDATEPROPORTION) CANDIDATEPROPORTION="$2"; shift ;;
        --HL_XP) HL_XP="$2"; shift ;;
        --STRAT) STRAT="'$2'"; shift ;;
        --NO_CV) NO_CV="$2"; shift ;;
        --K_TOP) K_TOP="$2"; shift ;;
        --SUBSET_RAND_CANDIDAT) SUBSET_RAND_CANDIDAT="$2"; shift ;;
        --HL_MAX_EPOCH) HL_MAX_EPOCH="$2"; shift ;;
        --CURRICULUM) CURRICULUM="$2"; shift ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
    shift
done

# ======================
# 4. Construction de la commande
# ======================
CMD="python RunSimulation.py \
--Data $DATA \
--TaskID $TASKID \
--NReplications $NREPLICATIONS \
--CandidateProportion $CANDIDATEPROPORTION \
--hl_xp $HL_XP \
--strat $STRAT \
--k_top $K_TOP \
--hl_max_epoch $HL_MAX_EPOCH"

# Ajout conditionnel de --subset_rand_candidat (uniquement si SUBSET_RAND_CANDIDAT != -1)
if [ "$SUBSET_RAND_CANDIDAT" -ne -1 ]; then
    CMD="$CMD --subset_rand_candidat $SUBSET_RAND_CANDIDAT"
fi

# Ajout conditionnel de --no_cv et --curriculum
[ "$NO_CV" -eq 1 ] && CMD="$CMD --no_cv"
[ "$CURRICULUM" -eq 1 ] && CMD="$CMD --curriculum"

# ======================
# 5. Génération du nom du job SLURM
# ======================
# Extraire tout ce qui suit "--k_top"
base_name=$(echo "$CMD" | sed -n 's/.*--k_top //p')

# Supprimer les espaces et caractères non-safe
base_name=$(echo "$base_name" | tr -d ' ' | tr -cd '[:alnum:]_.-')

# Ajouter un timestamp pour éviter les conflits
job_name="${base_name}_${DATA}_T${TASKID}_${STRAT}_$(date +%Y%m%d_%H%M%S)"

echo "Nom du job SLURM : $job_name"

# ======================
# 6. Génération du fichier .slurm
# ======================
CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda_wigs

cat > "temp_${job_name}.slurm" <<EOL_
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.out
#SBATCH --error=logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --account hir@h100
#SBATCH -C h100
#SBATCH --time=1:50:00
#SBATCH --qos=qos_gpu_h100-dev

CONDA_PATH_INSTALL=$CONDA_PATH_INSTALL

module purge
module load arch/h100
module load miniforge/24.11.3
conda activate pytorch-gpu-2.8.0+py3.12.11
conda activate --stack \$CONDA_PATH_INSTALL
module load pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=\$CONDA_PATH_INSTALL

$CMD
EOL_

# ======================
# 7. Affichage et soumission
# ======================
echo "Commande exécutée :"
echo "$CMD"
echo ""

# Créer le dossier logs s'il n'existe pas
mkdir -p logs

# Soumettre le job
sbatch "temp_${job_name}.slurm"

# Supprimer le fichier temporaire
rm "temp_${job_name}.slurm"