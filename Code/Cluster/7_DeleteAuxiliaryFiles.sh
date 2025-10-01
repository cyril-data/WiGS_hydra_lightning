#!/bin/bash

echo "--- Starting Auxiliary File Cleanup ---"

### Define Project Paths ###
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CODE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$CODE_DIR/..")

### Define Directories ###
SBATCH_DIR="$CODE_DIR/Cluster/RunSimulations"
LOG_DIR="$CODE_DIR/Cluster/RunSimulations/ClusterMessages"

### Delete .sbatch files ###
find "$SBATCH_DIR" -name "master_job_*.sbatch" -delete
echo "Deleted .sbatch files."

### Delete .err and .out files ###
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR/out" -name "*.out" -delete
    find "$LOG_DIR/error" -name "*.err" -delete
    echo "Deleted .out and .err log files."
else
    echo "Log directory not found. Skipping."
fi

echo "--- Auxiliary cleanup completed ---"