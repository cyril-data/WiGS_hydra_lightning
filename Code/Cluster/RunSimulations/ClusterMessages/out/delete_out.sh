#!/bin/bash

# Function to delete all out files
delete_out_files() {
    # Check for out files in the current directory
    if ls *.out 1> /dev/null 2>&1; then
        rm *.out
        echo "All .out files deleted."
    else
        echo "No .out files found in $(pwd)."
    fi
}

# Execute the function when the script is run
delete_out_files
