#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DIR"

cd ~/WeightedGreedySampling
python Code/utils/Auxiliary/GenerateJobs.py