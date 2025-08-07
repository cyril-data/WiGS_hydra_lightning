#!/bin/bash

### Get the current directory name ###
CURRENT_DIR=$(basename "$PWD")
echo "Current directory is: $CURRENT_DIR"

### Navigate to RunSimulations Directory ###
cd RunSimulations

### Delete all .sbatch files ###
if [ -f delete_sbatch.sh ]; then
    bash delete_sbatch.sh
else
    echo "No delete_sbatch.sh script found."
fi

### Delete all .out files ###
if [ -d ClusterMessages/out ]; then
    cd ClusterMessages/out
    if [ -f delete_out.sh ]; then
        bash delete_out.sh
    else
        # Check if there are any .out files
        if ls *.out 1> /dev/null 2>&1; then
            rm *.out
            echo "All .out files deleted."
        else
            echo "No .out files found in $(pwd)."
        fi
    fi
    
    ### Delete all .err files ###
    if [ -d ../error ]; then
        cd ../error
        if [ -f delete_err.sh ]; then
            bash delete_err.sh
        else
            # Check if there are any .err files
            if ls *.err 1> /dev/null 2>&1; then
                rm *.err
                echo "All .err files deleted."
            else
                echo "No .err files found in $(pwd)."
            fi
        fi
    else
        echo "No error directory found in ClusterMessages."
    fi
else
    echo "No ClusterMessages/out directory found."
fi

echo "Cleanup completed."