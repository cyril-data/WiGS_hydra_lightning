#!/bin/bash

### Get the current directory name (e.g., BankNote) ###
CURRENT_DATASET=$(basename "$PWD")
echo "Processing delete for dataset: $CURRENT_DATASET"

# Define the root results directory
RESULTS_ROOT_DIR="$HOME/RashomonActiveLearning/Results"

# Define the list 
MODEL_DIRS=(
    "BayesianNeuralNetworkPredictor"
    "GaussianProcessClassifierPredictor"
    "RandomForestClassifierPredictor"
    "TreeFarmsPredictor"   
    "LFRPredictor" 
)

echo "--- Starting Raw Results Deletion for $CURRENT_DATASET ---"

# Loop through each new model directory name
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    TARGET_RAW_DIR="$RESULTS_ROOT_DIR/$CURRENT_DATASET/$MODEL_DIR/Raw"

    echo "Attempting to delete .pkl files in: $TARGET_RAW_DIR"

    # Check if the target Raw directory exists
    if [ -d "$TARGET_RAW_DIR" ]; then
        # Navigate into the Raw directory. Using pushd/popd to manage directory stack.
        pushd "$TARGET_RAW_DIR" > /dev/null || { echo "Error: Could not change directory to $TARGET_RAW_DIR"; exit 1; }

        # Execute the delete_results.sh script within that Raw directory
        if [ -f "./delete_results.sh" ]; then
            bash ./delete_results.sh
            # The delete_results.sh should ideally output confirmation itself
        else
            echo "Warning: './delete_results.sh' not found in $TARGET_RAW_DIR. Attempting 'rm *.pkl' directly."
            # Check if there are any .pkl files before trying to remove them to avoid 'No such file' error
            if ls *.pkl 1> /dev/null 2>&1; then
                rm *.pkl
                echo "All .pkl files deleted from $TARGET_RAW_DIR."
            else
                echo "No .pkl files found in $TARGET_RAW_DIR."
            fi
        fi
        
        # Navigate back to the previous directory (where pushd was called from)
        popd > /dev/null
        
    else
        echo "Skipping: Model directory '$MODEL_DIR' not found for dataset '$CURRENT_DATASET' at path: $TARGET_RAW_DIR"
    fi
done

echo "--- Deletion of Raw Results Completed for $CURRENT_DATASET ---"
