#!/bin/bash

#SBATCH --job-name=GenPlotsTrends
#SBATCH --partition=short
#SBATCH --array=1-20  
#SBATCH --output=RunSimulations/ClusterMessages/out/genplots_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/genplots_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=5GB

### Script Execution ###
echo "--- Starting PLOT and TREND generation for Task ${SLURM_ARRAY_TASK_ID} ---"

### 1. Get Project Root and CD ###
cd "$SLURM_SUBMIT_DIR/../.."
echo "Current Directory: $(pwd)"
PROJECT_ROOT=$(pwd)
CODE_DIR="${PROJECT_ROOT}/Code"

### 2. Get This Job's Dataset ###
AGG_DIR="Results/simulation_results/aggregated"
ALL_DATASETS=($(find "$AGG_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))
NUM_DATASETS=${#ALL_DATASETS[@]}

echo "Found ${NUM_DATASETS} total datasets."
echo "NOTE: SLURM --array flag should be set to 1-${NUM_DATASETS}"

### 3. Get This Job's Dataset ###
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_TO_PROCESS=${ALL_DATASETS[$task_id_zero_based]}

if [ -z "$DATASET_TO_PROCESS" ]; then
    echo "Error: Task ID ${SLURM_ARRAY_TASK_ID} is out of bounds. No dataset found."
    exit 1
fi

echo "Processing dataset: ${DATASET_TO_PROCESS}"

# ======================================================
# --- PART 1: Generate Trace Plots ---
# ======================================================
python3 "${CODE_DIR}/utils/Auxiliary/GeneratePlots.py" --dataset "$DATASET_TO_PROCESS" --no-legend
echo "--- Trace plots finished for ${DATASET_TO_PROCESS} ---"


# ======================================================
# --- PART 2: Generate Weight Trends ---
# ======================================================
echo "--- Starting Weight Trend Generation for: ${DATASET_TO_PROCESS} ---"

### Define Parameters ###
SELECTOR_FOR_TRENDS="WiGS (SAC)"
SEED_TO_PLOT_INDIV= ("0" "1" "2")

IMG_AVG_TRENDS_DIR="Results/images/manuscript/average_weight_trends"
APP_INDIV_TRENDS_DIR="Results/images/appendices/individual_weight_trends"

# Create directories 
mkdir -p "${IMG_AVG_TRENDS_DIR}"
mkdir -p "${APP_INDIV_TRENDS_DIR}"

### Run the generation script ###
exact_weight_file="Results/simulation_results/aggregated/${DATASET_TO_PROCESS}/weight_history/${SELECTOR_FOR_TRENDS}_WeightHistory.csv"

if [ -f "$exact_weight_file" ]; then
    ## A. Generate AVERAGE Trend ##
    echo "  Processing Trend (Average) for: ${DATASET_TO_PROCESS}"
    python3 "${CODE_DIR}/utils/Auxiliary/AnalyzeWeightTrends.py" \
        --dgp_name "${DATASET_TO_PROCESS}" \
        --selector "${SELECTOR_FOR_TRENDS}" \
        --seed "all" \
        --output_dir "${IMG_AVG_TRENDS_DIR}"

    ## B. Generate INDIVIDUAL Seed Trends ##
    for seed in ${SEED_TO_PLOT_INDIV}; do
         echo "  Processing Trend (Seed ${seed}) for: ${DATASET_TO_PROCESS}"
         python3 "${CODE_DIR}/utils/Auxiliary/AnalyzeWeightTrends.py" \
            --dgp_name "${DATASET_TO_PROCESS}" \
            --selector "${SELECTOR_FOR_TRENDS}" \
            --seed "${seed}" \
            --output_dir "${APP_INDIV_TRENDS_DIR}"
    done
else
    echo "  Skipping Trend: ${DATASET_TO_PROCESS} / ${SELECTOR_FOR_TRENDS} (Weight file not found)"
fi

echo "--- Finished Weight Trend Generation for: ${DATASET_TO_PROCESS} ---"

# ======================================================
# --- PART 3: Generate Wilcoxon Stat-Test ---
# ======================================================
echo "--- Starting Wilcoxon Test for: ${DATASET_TO_PROCESS} ---"

TABLES_DIR="Results/tables"
mkdir -p "${TABLES_DIR}"

python3 "${CODE_DIR}/utils/Auxiliary/WilcoxonRankSignedTest.py" \
    --dataset "${DATASET_TO_PROCESS}" \
    --metric "RMSE"

echo "--- Wilcoxon test finished for ${DATASET_TO_PROCESS} ---"
echo "--- Finished All Tasks for ${SLURM_ARRAY_TASK_ID} ---"