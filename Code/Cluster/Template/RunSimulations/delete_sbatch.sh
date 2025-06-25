#!/bin/bash

# Function to delete all sbatch files
delete_sbatch_files() {
    # Check for sbatch files in the current directory
    if ls *.sbatch 1> /dev/null 2>&1; then
        rm *.sbatch
        echo "All .sbatch files deleted."
    else
        echo "No .sbatch files found in $(pwd)."
    fi
}

# Execute the function when the script is run
delete_sbatch_files
