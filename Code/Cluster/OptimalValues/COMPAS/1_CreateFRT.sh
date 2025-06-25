#!/bin/bash

### Current Directory Name ###
CURRENT_DIR=$(basename "$PWD")
echo "Create FindRashomonThreshold .sbatch files: $CURRENT_DIR"

cd ~/RashomonActiveLearning
python Code/utils/Auxiliary/CreateFRT.py \
    --DataType "$CURRENT_DIR" \
    --RashomonThreshold 0.05 \
