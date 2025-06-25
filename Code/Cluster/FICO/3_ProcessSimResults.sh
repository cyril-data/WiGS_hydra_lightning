#!/bin/bash

# This script is designed to be run from your dataset-specific cluster directory, e.g.,
# ~/RashomonActiveLearning/Code/Cluster/BankNote/
# It then navigates to the project root to run the Python script.

### Get Current Directory Name (e.g., BankNote or Iris) ###
CURRENT_DATASET=$(basename "$PWD")
echo "Processing results for dataset: $CURRENT_DATASET"

# Navigate to the project root directory
# Adjust this path if your project root is not directly at ~/RashomonActiveLearning
cd "$HOME/RashomonActiveLearning" || { echo "Error: Could not navigate to project root."; exit 1; }

# Define the path to the Python aggregation script
PROCESS_SCRIPT="Code/utils/Auxiliary/ProcessSimulationResults.py"

echo "--- Extracting Results for $CURRENT_DATASET ---"

# The ModelType argument must match the exact directory name under Results/<DATASET>/
# The Categories argument must be a unique substring that matches the JobName suffix in the .pkl files.

# 1. RF_PL: RandomForestClassifierPredictor + PassiveLearningSelector
#    Example JobName: 0IS_RF_PL_B1.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "RandomForestClassifierPredictor" \
    --Categories "_RF_PL_"

# 2. GPC_PL: GaussianProcessClassifierPredictor + PassiveLearningSelector
#    Example JobName: 0IS_GPC_PL_B1_KTRBF_KLS1_KNU1.5.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "GaussianProcessClassifierPredictor" \
    --Categories "_GPC_PL_"

# 3. BNN_PL: BayesianNeuralNetworkPredictor + PassiveLearningSelector
#    Example JobName: 0IS_BNN_PL_B1_HS50_DR2_E100_LR001_BST32.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "BayesianNeuralNetworkPredictor" \
    --Categories "_BNN_PL_"

# 4. BNN_BALD: BayesianNeuralNetworkPredictor + BALDSelector
#    Example JobName: 0IS_BNN_BALD_B1_HS50_DR2_E100_LR001_BST32_K20.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "BayesianNeuralNetworkPredictor" \
    --Categories "_BNN_BALD_"

# 5. GPC_BALD: GaussianProcessClassifierPredictor + BALDSelector
#    Example JobName: 0IS_GPC_BALD_B1_KTRBF_KLS1_KNU1.5_K20.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "GaussianProcessClassifierPredictor" \
    --Categories "_GPC_BALD_"

# # 6. UNREAL: TreeFarmsPredictor + BatchQBCSelector + UniqueErrorsInput=1
# #    Example JobName: 0IS_UNREAL_UEI1A5_DW0_DEW0_B1.pkl
# python "$PROCESS_SCRIPT" \
#     --DataType "$CURRENT_DATASET" \
#     --ModelType "TreeFarmsPredictor" \
#     --Categories "_UNREAL_" # Match start of category including UEI1A

# # 7. DUREAL: TreeFarmsPredictor + BatchQBCSelector + UniqueErrorsInput=0
# #    Example JobName: 0IS_DUREAL_UEI0A5_DW0_DEW0_B1.pkl
# python "$PROCESS_SCRIPT" \
#     --DataType "$CURRENT_DATASET" \
#     --ModelType "TreeFarmsPredictor" \
#     --Categories "_DUREAL_" # Match start of category including UEI0A

# 8. RF_QBC: RandomForestClassifierPredictor + BatchQBCSelector
#    Example JobName: 0IS_RF_QBC_DW0_DEW0_B1.pkl
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "RandomForestClassifierPredictor" \
    --Categories "_RF_QBC_" # Match start of category including UEI0A (assuming default for RF_QBC)

# 9. UNREAL_LFR: LFRPredictor + BatchQBCSelector + UniqueErrorsInput=1
#    Example JobName: 0IS_Ulfr_A5_DW0_DEW0_B1.pkl (your tree output showed Ulfr)
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "LFRPredictor" \
    --Categories "_Ulfr_" # Match start of category including Ulfr and A for Adder

# 10. DUREAL_LFR: LFRPredictor + BatchQBCSelector + UniqueErrorsInput=0
#     Example JobName: 0IS_Dlfr_A5_DW0_DEW0_B1.pkl (your tree output showed Dlfr)
python "$PROCESS_SCRIPT" \
    --DataType "$CURRENT_DATASET" \
    --ModelType "LFRPredictor" \
    --Categories "_Dlfr_" # Match start of category including Dlfr and A for Adder

echo "--- All Extraction Commands Submitted ---"