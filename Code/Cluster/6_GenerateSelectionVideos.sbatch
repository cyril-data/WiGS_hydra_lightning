#!/bin/bash

# --- SBATCH Directives ---
#SBATCH --job-name=VidSelect
#SBATCH --partition=short
#SBATCH --output=RunSimulations/ClusterMessages/out/vidselect_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/vidselect_%A_%a.err
#SBATCH --time=01:00:00  
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9


# --- INPUTS TO CONFIGURE --- #
DATASET="dgp_two_regime"         
SEED_TO_VISUALIZE=0              
VIDEO_LENGTH_SECONDS=90          
OUTPUT_BASE_DIR="Results/visualizations" 
CLEANUP_FRAMES=true            

# --- Array of Selectors to Process --- #
declare -a SELECTORS_TO_PROCESS=(
    "Passive Learning"
    "GSx"
    "GSy"
    "iGS"
    "WiGS (Static w_x=0.25)"
    # "WiGS (Static w_x=0.5)" 
    "WiGS (Static w_x=0.75)"
    "WiGS (Time-Decay, Linear)"
    "WiGS (Time-Decay, Exponential)"
    # "WiGS (MAB-UCB1, c=0.5)" 
    # "WiGS (MAB-UCB1, c=2.0)" 
    "WiGS (MAB-UCB1, c=5.0)"
    "WiGS (SAC)"
)

# --- Calculate Array Size ---
NUM_SELECTORS=${#SELECTORS_TO_PROCESS[@]}
LAST_ARRAY_INDEX=$((NUM_SELECTORS - 1))

echo "#########################################################################"
echo "# REMINDER: Manually set or use a wrapper script to set the             #"
echo "# SBATCH --array directive below to: --array=0-${LAST_ARRAY_INDEX}           #"
echo "#########################################################################"

# --- Per-Task Execution (Each task handles one selector) ---
TASK_ID=$SLURM_ARRAY_TASK_ID
SELECTOR=${SELECTORS_TO_PROCESS[$TASK_ID]} 
if [ -z "$SELECTOR" ]; then
    echo "Error: Task ID $TASK_ID is out of bounds for selector array. Exiting."
    exit 1
fi

echo "--- Starting Task ${TASK_ID} for Selector: ${SELECTOR} ---"
PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
cd "$PROJECT_ROOT" 

# --- Calculate N_ITERATIONS for THIS selector ---
SAFE_SELECTOR=${SELECTOR// /_}
SAFE_SELECTOR=${SAFE_SELECTOR//(/}
SAFE_SELECTOR=${SAFE_SELECTOR//)/}
SAFE_SELECTOR=${SAFE_SELECTOR//,/}
SAFE_SELECTOR=${SAFE_SELECTOR//=/}
SAFE_SELECTOR=${SAFE_SELECTOR//:/}

HISTORY_FILE="Results/simulation_results/aggregated/${DATASET}/selection_history/${SAFE_SELECTOR}_SelectionHistory.csv"
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

# --- Call Python Visualization Script ---
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

echo "--- Finished Task ${TASK_ID} for Selector: ${SELECTOR} ---"