#!/bin/bash

### Get the current directory name ###
CURRENT_DIR=$(basename "$PWD")
echo "Current directory is: $CURRENT_DIR"

### Delete all Unprocessed Results files ##
cd ../../../../Results/OptimalThreshold/"$CURRENT_DIR"/Raw
bash delete_results.sh
echo "All .pkl results files in FRT deleted."

