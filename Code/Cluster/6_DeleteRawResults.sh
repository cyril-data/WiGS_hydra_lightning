#!/bin/bash

echo "--- Preparing to delete RAW simulation results ---"

### Define Project Paths ###
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CODE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$CODE_DIR/..")
RAW_RESULTS_DIR="$PROJECT_ROOT/Results/simulation_results/raw"

### Delete Raw Simulation Results ###
echo "Searching for raw result files in $RAW_RESULTS_DIR..."

if [ -d "$RAW_RESULTS_DIR" ]; then
    for dataset_dir in "$RAW_RESULTS_DIR"/*/; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            
            # Find and delete .pkl files only within this dataset's directory
            find "$dataset_dir" -name "*.pkl" -delete
            
            # Print the summary message for this dataset
            echo "Raw results files for $dataset_name deleted."
        fi
    done
else
    echo "Raw results directory not found. Skipping."
fi

echo "--- Raw results cleanup completed ---"