#!/bin/bash

echo "--- Generating All Key Manuscript Plots, Tables, and Legend ---"

### DIRECTORIESS ###
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CODE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$CODE_DIR/..")

cd "$PROJECT_ROOT"
echo "Changed working directory to ${PROJECT_ROOT}"

## Define input and output directories ##
AGG_RESULTS_DIR="Results/simulation_results/aggregated"
TABLES_DIR="Results/tables"
IMG_MANUSCRIPT_DIR="Results/images/manuscript"

## Define sub-directories ##
IMG_AVG_TRENDS_DIR="${IMG_MANUSCRIPT_DIR}/average_weight_trends"
APP_BASE_DIR="Results/images/appendices"
APP_INDIV_HEATMAPS_DIR="${APP_BASE_DIR}/individual_weight_heatmaps"
APP_INDIV_TRENDS_DIR="${APP_BASE_DIR}/individual_weight_trends"

## Create all output directories ##
mkdir -p "${TABLES_DIR}"
mkdir -p "${IMG_MANUSCRIPT_DIR}"
mkdir -p "${IMG_AVG_TRENDS_DIR}"      
mkdir -p "${APP_INDIV_HEATMAPS_DIR}" 
mkdir -p "${APP_INDIV_TRENDS_DIR}"   

### Define Key Parameters ###
declare -a SYNTHETIC_DGPS=("dgp_two_regime" "dgp_three_regime")
SELECTOR_FOR_HEATMAP="WiGS (SAC)"
SEED_TO_PLOT_INDIV=("0" "1" "2")

# ======================================================
# --- PART 1: Generate Core Manuscript Figures
# ======================================================

## Plot 1: Nearest Neighbor Visual ##
echo ""
echo "--- 1. Generating Nearest Neighbor plot... ---"
NEAREST_NEIGHBOR_PATH="${IMG_MANUSCRIPT_DIR}/NearestNeighborVisualization.png"
python3 "${CODE_DIR}/utils/Auxiliary/NearestNeighborVisualization.py" --output_file "${NEAREST_NEIGHBOR_PATH}"
echo "Nearest Neighbor plot saved to ${IMG_MANUSCRIPT_DIR}/"

## Plots 2 & 3: DGP Visualization ##
echo ""
echo "--- 2. Generating DGP visualization images... ---"
python3 "${CODE_DIR}/utils/Auxiliary/GenerateDGPImage.py" --output_dir "${IMG_MANUSCRIPT_DIR}"
echo "DGP images saved to ${IMG_MANUSCRIPT_DIR}/"

## Plots 4 & 5: Weight Heatmaps (Average and Single Seed) ##
echo ""
echo "--- 3. Generating Weight Heatmap plots for ${SELECTOR_FOR_HEATMAP}... ---"
for dgp in "${SYNTHETIC_DGPS[@]}"; do
    exact_weight_file="Results/simulation_results/aggregated/${dgp}/weight_history/${SELECTOR_FOR_HEATMAP}_WeightHistory.csv"
    if [ -f "$exact_weight_file" ]; then
        
        # A. Generate AVERAGE Heatmap #
        echo "  Processing Heatmap (Average) for: ${dgp}"
        python3 "${CODE_DIR}/utils/Auxiliary/PlotWeightHeatmap.py" \
            --dgp_name "${dgp}" \
            --selector "${SELECTOR_FOR_HEATMAP}" \
            --seed "avg" \
            --output_dir "${IMG_MANUSCRIPT_DIR}" 

        # B. Generate SINGLE SEED Heatmap #
        for seed in "${SEED_TO_PLOT_INDIV[@]}"; do
            echo "  Processing Heatmap (Seed ${seed}) for: ${dgp}"
            python3 "${CODE_DIR}/utils/Auxiliary/PlotWeightHeatmap.py" \
                --dgp_name "${dgp}" \
                --selector "${SELECTOR_FOR_HEATMAP}" \
                --seed "${seed}" \
                --output_dir "${APP_INDIV_HEATMAPS_DIR}" 
        done
    else
        echo "  Skipping Heatmaps: ${dgp} / ${SELECTOR_FOR_HEATMAP} (Weight file not found)"
    fi
done
echo "Weight Heatmap plots generated."

## Plot 6: Supporting Plot: Standalone Legend ##
echo ""
echo "--- 4. Generating Standalone Legend... ---"
python3 "${CODE_DIR}/utils/Auxiliary/GeneratePlots.py" --legend_only
mv "Results/images/benchmark_legend.png" "${IMG_MANUSCRIPT_DIR}/benchmark_legend.png"
echo "Legend saved to ${IMG_MANUSCRIPT_DIR}/"

echo ""
echo "--- ALL MANUSCRIPT PLOTS ARE GENERATED ---"