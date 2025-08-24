#!/bin/bash

echo "--- STARTING MASTER WORKFLOW ---"

# --- 1. Navigate to Project Root ---
# Get the directory where this .sh script is located (e.g., Code/Cluster)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Navigate to the main project root directory
cd "$SCRIPT_DIR/../../"

# --- 2. Set up Environment ---
echo "Activating Conda environment..."
# Now that we are in the project root, this command will work correctly
source ./.WiGS_Env/bin/activate

# --- 3. Run Initial Setup ---
echo -e "\nStep 1: Running Data Preprocessing..."
# Scripts are now called from the project root
./Code/Cluster/1_PreprocessData.sh

echo -e "\nStep 2: Generating SLURM job scripts..."
./Code/Cluster/2_CreateSimulationSbatch.sh

# --- 4. Submit Simulation Jobs ---
echo -e "\nStep 3: Submitting simulation job arrays to the queue..."
JOB_IDS=""
for f in Code/Cluster/RunSimulations/master_job_*.sbatch; do
    if [ -f "$f" ]; then
        JOB_ID=$(sbatch --parsable "$f")
        echo "  > Submitted $f with Job ID: $JOB_ID"
        if [ -z "$JOB_IDS" ]; then
            JOB_IDS="$JOB_ID"
        else
            JOB_IDS="$JOB_IDS:$JOB_ID"
        fi
    fi
done

if [ -z "$JOB_IDS" ]; then
    echo "No simulation jobs were submitted. Exiting."
    exit 1
fi

# --- 5. Submit Dependent Analysis & Cleanup Jobs ---
echo -e "\nSubmitting dependent jobs (aggregation, plotting, cleanup)..."
AGG_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_IDS Code/Cluster/4_ProcessResults.sh)
PLOT_JOB_ID=$(sbatch --parsable --dependency=afterok:$AGG_JOB_ID Code/Cluster/5_ImageGeneration.sh)
AUX_DEL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PLOT_JOB_ID Code/Cluster/6_DeleteAuxiliaryFiles.sh)
RAW_DEL_JOB_ID=$(sbatch --parsable --dependency=afterok:$AUX_DEL_JOB_ID Code/Cluster/7_DeleteRawResults.sh)
echo "  > Dependent jobs submitted."

echo -e "\n--- MASTER WORKFLOW SUBMITTED ---"
echo "Monitor progress with: squeue -u $USER"