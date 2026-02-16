#!/bin/bash

echo "--- Starting Data Preprocessing ---"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

echo "Running the data preprocessing script..."
python utils/Auxiliary/PreprocessData.py

echo "--- Preprocessing script finished. ---"