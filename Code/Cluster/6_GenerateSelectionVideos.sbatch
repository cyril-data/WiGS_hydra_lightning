#!/bin/bash
#SBATCH --job-name=VidSelect
#SBATCH --partition=short
#SBATCH --output=RunSimulations/ClusterMessages/out/vidselect_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/vidselect_%A_%a.err
#SBATCH --time=11:59:00        
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-25   

### INPUTS TO CONFIGURE ###
SEED_TO_VISUALIZE=0              
VIDEO_LENGTH_SECONDS=90          
OUTPUT_BASE_DIR="Results/selection_videos"
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
    "WiGS (Static w_x=0.5)"     
    "WiGS (Static w_x=0.75)"
    "WiGS (Time-Decay, Linear)"
    "WiGS (Time-Decay, Exponential)"
    "WiGS (MAB-UCB1, c=0.5)"     
    "WiGS (MAB-UCB1, c=2.0)"     
    "WiGS (MAB-UCB1, c=5.0)"
    "WiGS (SAC)"
)

### Calculate Array Size ###
NUM_SELECTORS=${#SELECTORS_TO_PROCESS[@]}
NUM_DATASETS=${#DATASETS_TO_PROCESS[@]}
TOTAL_JOBS=$((NUM_SELECTORS * NUM_DATASETS))
LAST_ARRAY_INDEX=$((TOTAL_JOBS - 1))

echo "#########################################################################"
echo "# Setup: ${NUM_DATASETS} Datasets x ${NUM_SELECTORS} Selectors"
echo "# Total jobs required: ${TOTAL_JOBS}. Array range should be: 0-${LAST_ARRAY_INDEX}"
echo "#########################################################################"

### Map 1D Task ID to 2D Dataset/Selector ###
TASK_ID=$SLURM_ARRAY_TASK_ID

# Safety check for array bounds
if [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Error: Task ID $TASK_ID exceeds total jobs ($TOTAL_JOBS). Check --array range."
    exit 1
fi

DATASET_INDEX=$((TASK_ID / NUM_SELECTORS))
SELECTOR_INDEX=$((TASK_ID % NUM_SELECTORS))

DATASET=${DATASETS_TO_PROCESS[$DATASET_INDEX]}
SELECTOR=${SELECTORS_TO_PROCESS[$SELECTOR_INDEX]}

echo "--- Starting Task ${TASK_ID}: ${DATASET} / ${SELECTOR} ---"
PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
cd "$PROJECT_ROOT" 

### Calculate N_ITERATIONS ###
HISTORY_FILE="Results/simulation_results/aggregated/${DATASET}/selection_history/${SELECTOR}_SelectionHistory.csv"

echo "DEBUG: Looking for history file at:"
echo "'$HISTORY_FILE'"

if [ -f "$HISTORY_FILE" ]; then
    LINE_COUNT=$(wc -l < "$HISTORY_FILE")
    N_ITERATIONS=$((LINE_COUNT - 2))
    
    if [ "$N_ITERATIONS" -lt 0 ]; then N_ITERATIONS=0; fi
    echo "Found file. Lines: $LINE_COUNT. Simulation Iterations: ${N_ITERATIONS}"
else
    echo "!!!!!!!! ERROR !!!!!!!!"
    echo "History file NOT found for ${SELECTOR}."
    echo "Expected path: $HISTORY_FILE"
    echo "Verify the filename matches exactly (including spaces/parentheses)."
    exit 1 
fi

### Prepare Conditional Flags ###
CMD_FLAGS=""
if [ "$CLEANUP_FRAMES" = true ]; then
    CMD_FLAGS="--cleanup_frames"
fi

### Call Python Visualization Script ###
echo "Running Python script..."
python Code/utils/Auxiliary/VisualizeSelections.py \
    --dgp_name "$DATASET" \
    --selector "$SELECTOR" \
    --seed "$SEED_TO_VISUALIZE" \
    --video_length "$VIDEO_LENGTH_SECONDS" \
    --output_dir "$OUTPUT_BASE_DIR" \
    --num_iterations "$N_ITERATIONS" \
    $CMD_FLAGS

# Check Python exit code
RET_CODE=$?
if [ $RET_CODE -ne 0 ]; then
  echo "Error: Python script failed for selector: ${SELECTOR} with error code $RET_CODE"
  exit 1 
fi

echo "--- Finished Task ${TASK_ID} Successfully ---"