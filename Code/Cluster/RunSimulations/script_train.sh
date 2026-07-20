#!/bin/bash

# ======================
# 1. Valeurs par défaut
# ======================
DATA='hydralightning'
TASKID=0
NREPLICATIONS=25
CANDIDATEPROPORTION=0.999
HL_XP='full_fold1_active_learning_h100'
NO_CV=1
STRAT='iGS'
HL_MAX_EPOCH=1
K_TOP=1000
SUBSET_RAND_CANDIDAT=-1
CURRICULUM=1
RES_FREQ=1
MACHINE='h100'
TIME=20
PROJET='hir'
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
        --RES_FREQ) RES_FREQ="$2"; shift ;;
        --MACHINE) MACHINE="$2"; shift ;;
        --TIME) TIME="$2"; shift ;;
        --PROJET) PROJET="$2"; shift ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
    shift
done

# ======================
# 3. Modify Slurm and CMD arg
# ======================

MODULE_LOAD_H100=''

# Adapt qos and cpus-per-task with MACHINE
if [[ "$MACHINE" == "h100" ]]; then
    ACCOUNT=$MACHINE
    CPU_PER_TASK=24
    MODULE_LOAD_H100='module load arch/h100'
    if [[ $TIME -le 2 ]]; then
        QOS="qos_gpu_h100-dev"
    elif [[ $TIME -le 20 ]]; then
        QOS="qos_gpu_h100-t3"
    else
        QOS="qos_gpu_h100-t4"
    fi
elif [[ "$MACHINE" == "a100" ]]; then
    ACCOUNT=$MACHINE
    CPU_PER_TASK=8
    if [[ $TIME -le 2 ]]; then
        QOS="qos_gpu_a100-dev"
    elif [[ $TIME -le 20 ]]; then
        QOS="qos_gpu_a100-t3"
    else
        echo "ERROR: a100 can not exceed 20h"
        exit 1
    fi
elif [[ "$MACHINE" == "v100" ]]; then
    ACCOUNT=$MACHINE
    MACHINE='v100-32g'
    CPU_PER_TASK=10
    if [[ $TIME -le 2 ]]; then
        QOS="qos_gpu-dev"
    elif [[ $TIME -le 20 ]]; then
        QOS="qos_gpu-t3"
    else
        QOS="qos_gpu-t4"
    fi
else
    echo "ERROR: Unknown MACHINE type"
    exit 1
fi

# Reduce TIME of 10 min
TIME=$((TIME - 1))
TIME=${TIME}:50:00



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
--res_freq $RES_FREQ \
--hl_max_epoch $HL_MAX_EPOCH \
--hl_worker $CPU_PER_TASK"


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
job_name="${MACHINE}_${TIME}_${HL_XP}_${base_name}_${STRAT}_$(date +%Y%m%d_%H%M%S)"

# ======================
# 6. Génération du fichier .slurm
# ======================
CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda_wigs


echo ""
echo "   --- Slurm config ---"
echo "QOS=$QOS"
echo "MACHINE=$MACHINE"
echo "ACCOUNT=$ACCOUNT"
echo "CPU_PER_TASK=$CPU_PER_TASK"
echo "TIME=$TIME"
echo "MODULE_LOAD_H100=$MODULE_LOAD_H100"
echo "job_name=${job_name}"
echo ""

cat > "temp_${job_name}.slurm" <<EOL_
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.out
#SBATCH --error=logs/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPU_PER_TASK}
#SBATCH --hint=nomultithread
#SBATCH --account ${PROJET}@${ACCOUNT}
#SBATCH -C ${MACHINE}
#SBATCH --time=${TIME}
#SBATCH --qos=${QOS}

CONDA_PATH_INSTALL=$CONDA_PATH_INSTALL

module purge
${MODULE_LOAD_H100}
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
echo "   --- Commande exécutée : ---"
echo "$CMD"
echo ""

# Créer le dossier logs s'il n'existe pas
mkdir -p logs

# Soumettre le job
sbatch "temp_${job_name}.slurm"

# Supprimer le fichier temporaire
rm "temp_${job_name}.slurm"