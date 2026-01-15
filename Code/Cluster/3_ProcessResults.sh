#!/bin/bash

### Set up ###
echo "--- Aggregating raw results ---"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

### Run script ###
python utils/Auxiliary/AggregateResults.py