#!/bin/bash
#SBATCH --job-name=VidSelect
#SBATCH --partition=short
#SBATCH --output=RunSimulations/ClusterMessages/out/vidselect_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/vidselect_%A_%a.err
#SBATCH --time=04:00:00        
#SBATCH --mem-per-cpu=5GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19          


### INPUTS TO CONFIGURE ###
SEED_TO_VISUALIZE=0              
VIDEO_LENGTH_SECONDS=90          
OUTPUT_BASE_DIR="Results/images/appendices/selection_videos"
CLEANUP_FRAMES=true            

declare -a DATASETS_TO_PROCESS=(
    "dgp_two_regime"
    "dgp_three_regime"
)
declare -a SELECTORS_TO_PROCESS=(
    "Passive Learning"
    "GSx"
    "GSy"
    "iGS"
    "WiGS (Static w_x=0.25)"
    "WiGS (Static w_x=0.75)"
    "WiGS (Time-Decay, Linear)"
    "WiGS (Time-Decay, Exponential)"
    "WiGS (MAB-UCB1, c=5.0)"
    "WiGS (SAC)"
)

### Calculate Array Size ###
NUM_SELECTORS=${#SELECTORS_TO_PROCESS[@]}
NUM_DATASETS=${#DATASETS_TO_PROCESS[@]}
TOTAL_JOBS=$((NUM_SELECTORS * NUM_DATASETS))
LAST_ARRAY_INDEX=$((TOTAL_JOBS - 1))

echo "#########################################################################"
echo "# This is a 2D job array for ${NUM_DATASETS} datasets and ${NUM_SELECTORS} selectors."
echo "# Total jobs: ${TOTAL_JOBS}. Array range: 0-${LAST_ARRAY_INDEX}"
echo "#########################################################################"

### Map 1D Task ID to 2D Dataset/Selector ###
TASK_ID=$SLURM_ARRAY_TASK_ID
DATASET_INDEX=$((TASK_ID / NUM_SELECTORS))
SELECTOR_INDEX=$((TASK_ID % NUM_SELECTORS))

DATASET=${DATASETS_TO_PROCESS[$DATASET_INDEX]}
SELECTOR=${SELECTORS_TO_PROCESS[$SELECTOR_INDEX]}

if [ -z "$SELECTOR" ] || [ -z "$DATASET" ]; then
    echo "Error: Task ID $TASK_ID is out of bounds. Exiting."
    exit 1
fi

echo "--- Starting Task ${TASK_ID} for: ${DATASET} / ${SELECTOR} ---"
PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
cd "$PROJECT_ROOT" 

### Calculate N_ITERATIONS for THIS selector ###
HISTORY_FILE="Results/simulation_results/aggregated/${DATASET}/selection_history/${SELECTOR}_SelectionHistory.csv"
echo "Looking for history file: $HISTORY_FILE"

N_ITERATIONS=0
if [ -f "$HISTORY_FILE" ]; then
    N_ITERATIONS=$(($(wc -l < "$HISTORY_FILE") - 2)) 
    if [ "$N_ITERATIONS" -lt 0 ]; then N_ITERATIONS=0; fi
    echo "Found ${N_ITERATIONS} iterations (0 to ${N_ITERATIONS})."
else
    echo "Warning: History file not found for ${SELECTOR}. Cannot generate video."
    exit 0
fi

### Call Python Visualization Script ###
echo "Running VisualizeSelections.py for ${SELECTOR}, Seed ${SEED_TO_VISUALIZE}"
python Code/utils/Auxiliary/VisualizeSelections.py \
    --dgp_name "$DATASET" \
    --selector "$SELECTOR" \
    --seed "$SEED_TO_VISUALIZE" \
    --video_length "$VIDEO_LENGTH_SECONDS" \
    --output_dir "$OUTPUT_BASE_DIR" \
    --num_iterations "$N_ITERATIONS" \
    --cleanup_frames

if [ $? -ne 0 ]; then
  echo "Error: Python script failed for selector: ${SELECTOR}"
  exit 1 
fi

echo "--- Finished Task ${TASK_ID} for: ${DATASET} / ${SELECTOR} ---"