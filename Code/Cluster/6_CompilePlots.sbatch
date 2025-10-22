#!/bin/bash

#SBATCH --job-name=CompilePlots
#SBATCH --partition=short
#SBATCH --array=1-8
#SBATCH --output=RunSimulations/ClusterMessages/out/compile_%A_%a.out
#SBATCH --error=RunSimulations/ClusterMessages/error/compile_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=2GB

### INPUTS ###
COLUMNS=2
ROWS=2
EVAL_TYPES=("full_pool")
PLOT_TYPES=("trace" "trace_relative_iGS")
METRICS=("RMSE" "MAE" "R2" "CC")

# --- Use the SLURM_ARRAY_TASK_ID to select a unique combination ---
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
eval_idx=$((task_id_zero_based / 8))
plot_idx=$(((task_id_zero_based % 8) / 4))
metric_idx=$((task_id_zero_based % 4))
EVAL_TYPE=${EVAL_TYPES[$eval_idx]}
PLOT_TYPE=${PLOT_TYPES[$plot_idx]}
METRIC=${METRICS[$metric_idx]}

### --- Script Execution --- ###
echo "--- Starting plot compilation for Task ${SLURM_ARRAY_TASK_ID} ---"
cd "$SLURM_SUBMIT_DIR/../.."

echo "Current Directory: $(pwd)"
echo "Compiling: ${EVAL_TYPE} / ${PLOT_TYPE} / ${METRIC}"

python Code/utils/Auxiliary/CompilePlots.py \
    --eval_type "$EVAL_TYPE" \
    --metric "$METRIC" \
    --plot_type "$PLOT_TYPE" \
    --columns "$COLUMNS" \
    --rows "$ROWS"

echo "--- Finished Task ${SLURM_ARRAY_TASK_ID} ---"