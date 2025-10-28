#!/bin/bash

echo "--- Starting DGP and Weight Trend Visualization Generation ---"

# --- Define Project Paths (Relative to Cluster/ script location) ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CODE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$CODE_DIR/..")
cd "$PROJECT_ROOT"
VIS_OUTPUT_DIR="${PROJECT_ROOT}/Results/visualizations"
AGG_RESULTS_DIR="${PROJECT_ROOT}/Results/simulation_results/aggregated"

# --- 1. Generate DGP Images ---
echo ""
echo "Generating DGP visualization images..."
python "${CODE_DIR}/utils/Auxiliary/GenerateDGPImage.py"
echo "DGP images generated."

# --- 2. Generate Weight Trend Plots ---
echo ""
echo "Generating Weight Trend plots..."

# --- Configure which trends to plot ---
declare -a dgp_names=("dgp_two_regime" "dgp_three_regime")
declare -a selectors=("WiGS (SAC)" "WiGS (MAB-UCB1, c=5.0)")
declare -a seeds_to_plot=("0" "all")

# --- Loop and generate plots ---
for dgp in "${dgp_names[@]}"; do
    for selector in "${selectors[@]}"; do
        exact_weight_file="${AGG_RESULTS_DIR}/${dgp}/weight_history/${selector}_WeightHistory.csv"
        if [ -f "$exact_weight_file" ]; then
            for seed in "${seeds_to_plot[@]}"; do
                echo "    Generating: ${dgp} / ${selector} / Seed(s): ${seed}"
                python "${CODE_DIR}/utils/Auxiliary/AnalyzeWeightTrends.py" \
                    --dgp_name "${dgp}" \
                    --selector "${selector}" \
                    --seed "${seed}" \
                    --output_dir "${VIS_OUTPUT_DIR}"
            done
        else
             echo "  Skipping: ${dgp} / ${selector} (Weight file not found at '${exact_weight_file}')"
        fi
    done
done
echo "Weight Trend plots generated."


echo ""
echo "--- Visualization Generation Complete ---"