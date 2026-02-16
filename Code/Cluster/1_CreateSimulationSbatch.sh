#!/bin/bash

echo "--- Preparing to generate cluster jobs ---"

### Directory ####
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
PROJECT_ROOT=$(realpath "$PWD/..")
mkdir -p Cluster/RunSimulations/ClusterMessages/{out,error}
mkdir -p "$PROJECT_ROOT/Results/simulation_results/raw"
mkdir -p "$PROJECT_ROOT/Results/simulation_results/aggregated"
mkdir -p "$PROJECT_ROOT/Results/images"

### Run script ###
python Cluster/CreateSimulationSbatch.py
echo "--- Job generation complete. ---"