#!/bin/bash

### Set Up ###
echo "--- Starting analysis and plot generation ---"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

### Generate Plots ###
echo "Running GeneratePlots.py..."
python utils/Auxiliary/GeneratePlots.py
