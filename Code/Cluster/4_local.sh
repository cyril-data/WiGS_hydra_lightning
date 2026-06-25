#!/bin/bash

echo "--- Starting PLOT and TREND generation for all datasets ---"
PROJECT_ROOT=$(pwd)
CODE_DIR="${PROJECT_ROOT}/Code"
AGG_DIR="Results/simulation_results/aggregated"

### 1. Discover all datasets ###
ALL_DATASETS=($(find "$AGG_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))
NUM_DATASETS=${#ALL_DATASETS[@]}
echo "Found ${NUM_DATASETS} total datasets."

### 2. Loop through all datasets ###
for DATASET_TO_PROCESS in "${ALL_DATASETS[@]}"; do
    echo "Processing dataset: ${DATASET_TO_PROCESS}"

    # ======================================================
    # --- PART 1: Generate Trace Plots ---
    # ======================================================
    python3 "${CODE_DIR}/utils/Auxiliary/GeneratePlots.py" --dataset "$DATASET_TO_PROCESS"
    echo "--- Trace plots finished for ${DATASET_TO_PROCESS} ---"

    # ======================================================
    # --- PART 2: Generate Weight Trends ---
    # ======================================================
    echo "--- Starting Weight Trend Generation for: ${DATASET_TO_PROCESS} ---"
    SELECTOR_FOR_TRENDS="WiGS (SAC)"
    SEED_TO_PLOT_INDIV=("0" "1" "2")

    IMG_AVG_TRENDS_DIR="Results/images/manuscript/average_weight_trends"
    APP_INDIV_TRENDS_DIR="Results/images/appendices/individual_weight_trends"

    mkdir -p "${IMG_AVG_TRENDS_DIR}"
    mkdir -p "${APP_INDIV_TRENDS_DIR}"

    exact_weight_file="Results/simulation_results/aggregated/${DATASET_TO_PROCESS}/weight_history/${SELECTOR_FOR_TRENDS}_WeightHistory.csv"

    if [ -f "$exact_weight_file" ]; then
        ## A. Generate AVERAGE Trend ##
        echo "  Processing Trend (Average) for: ${DATASET_TO_PROCESS}"
        python3 "${CODE_DIR}/utils/Auxiliary/AnalyzeWeightTrends.py" \
            --dgp_name "${DATASET_TO_PROCESS}" \
            --selector "${SELECTOR_FOR_TRENDS}" \
            --seed "all" \
            --output_dir "${IMG_AVG_TRENDS_DIR}"

        ## B. Generate INDIVIDUAL Seed Trends ##
        for seed in ${SEED_TO_PLOT_INDIV}; do
             echo "  Processing Trend (Seed ${seed}) for: ${DATASET_TO_PROCESS}"
             python3 "${CODE_DIR}/utils/Auxiliary/AnalyzeWeightTrends.py" \
                --dgp_name "${DATASET_TO_PROCESS}" \
                --selector "${SELECTOR_FOR_TRENDS}" \
                --seed "${seed}" \
                --output_dir "${APP_INDIV_TRENDS_DIR}"
        done
    else
        echo "  Skipping Trend: ${DATASET_TO_PROCESS} / ${SELECTOR_FOR_TRENDS} (Weight file not found)"
    fi

    echo "--- Finished Weight Trend Generation for: ${DATASET_TO_PROCESS} ---"

    # ======================================================
    # --- PART 3: Generate Wilcoxon Stat-Test ---
    # ======================================================
    echo "--- Starting Wilcoxon Test for: ${DATASET_TO_PROCESS} ---"
    TABLES_DIR="Results/tables"
    mkdir -p "${TABLES_DIR}"

    python3 "${CODE_DIR}/utils/Auxiliary/WilcoxonRankSignedTest.py" \
        --dataset "${DATASET_TO_PROCESS}" \
        --metric "RMSE"

    echo "--- Wilcoxon test finished for ${DATASET_TO_PROCESS} ---"
done

echo "--- Finished All Tasks for all datasets ---"