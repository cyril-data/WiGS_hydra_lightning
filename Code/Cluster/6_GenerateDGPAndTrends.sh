#!/bin/bash

echo "--- Starting DGP, Trends, and Heatmap Visualization Generation ---"

# --- Define Project Paths (Relative to Cluster/ script location) ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CODE_DIR=$(realpath "$SCRIPT_DIR/..")
PROJECT_ROOT=$(realpath "$CODE_DIR/..")
VIS_OUTPUT_DIR="${PROJECT_ROOT}/Results/visualizations" # Base output dir for trends & heatmaps
DOCS_IMG_DIR="${PROJECT_ROOT}/docs/images"             # Output dir for DGP images
AGG_RESULTS_DIR="${PROJECT_ROOT}/Results/simulation_results/aggregated" # Path to aggregated results

# --- Ensure Python environment is active (optional but good practice) ---
# source "${PROJECT_ROOT}/.WiGS_Env/bin/activate" # Uncomment and adjust if needed

# --- 1. Generate DGP Images ---
echo ""
echo "Generating DGP visualization images..."
# Make sure the target directory exists
mkdir -p "${DOCS_IMG_DIR}"
python "${CODE_DIR}/utils/Auxiliary/GenerateDGPImage.py"
# (Adjust GenerateDGPImage.py if it doesn't save to DOCS_IMG_DIR by default)
echo "DGP images generated."

# --- 2. Generate Weight Trend Plots ---
echo ""
echo "Generating Weight Trend plots..."

# --- Configure which trends/heatmaps to plot ---
declare -a dgp_names=("dgp_two_regime" "dgp_three_regime")
# Selectors that actually produce weights
declare -a weighted_selectors=(
    "WiGS (Static w_x=0.25)"
    # "WiGS (Static w_x=0.5)" # Example: Skip
    "WiGS (Static w_x=0.75)"
    "WiGS (Time-Decay, Linear)"
    "WiGS (Time-Decay, Exponential)"
    "WiGS (MAB-UCB1, c=5.0)"
    "WiGS (SAC)"
)
# Specify which seed(s) to plot individually for trends/heatmaps
declare -a seeds_to_plot_individually=("0") # Just seed 0 for example
# Specify if you want the average trend plot (use "all" for the --seed arg)
generate_average_trend=true

# --- Loop for Trends ---
for dgp in "${dgp_names[@]}"; do
    for selector in "${weighted_selectors[@]}"; do
        # Construct the EXACT expected weight filename
        exact_weight_file="${AGG_RESULTS_DIR}/${dgp}/weight_history/${selector}_WeightHistory.csv"

        if [ -f "$exact_weight_file" ]; then
            echo "  Processing Trends for: ${dgp} / ${selector}"
            # Plot individual seeds
            for seed in "${seeds_to_plot_individually[@]}"; do
                echo "    Generating Trend Plot - Seed: ${seed}"
                python "${CODE_DIR}/utils/Auxiliary/VisualizeWeightTrends.py" \
                    --dgp_name "${dgp}" \
                    --selector "${selector}" \
                    --seed "${seed}" \
                    --output_dir "${VIS_OUTPUT_DIR}"
            done
            # Plot average if requested
            if [ "$generate_average_trend" = true ]; then
                echo "    Generating Trend Plot - Average (all seeds)"
                python "${CODE_DIR}/utils/Auxiliary/VisualizeWeightTrends.py" \
                    --dgp_name "${dgp}" \
                    --selector "${selector}" \
                    --seed "all" \
                    --output_dir "${VIS_OUTPUT_DIR}"
            fi
        else
             echo "  Skipping Trends: ${dgp} / ${selector} (Weight file not found)"
        fi
    done
done
echo "Weight Trend plots generated."

# --- 3. Generate Weight Heatmaps ---
echo ""
echo "Generating Weight Heatmap plots..."

# --- Loop for Heatmaps (only individual seeds make sense here) ---
for dgp in "${dgp_names[@]}"; do
    for selector in "${weighted_selectors[@]}"; do
        # Check if necessary files exist before trying to plot
        initial_file="${AGG_RESULTS_DIR}/${dgp}/InitialIndices.csv"
        # Need to handle potential spaces/parens in selector for selection file name
        safe_selector_name_hist=${selector// /_}
        safe_selector_name_hist=${safe_selector_name_hist//(/}
        safe_selector_name_hist=${safe_selector_name_hist//)/}
        safe_selector_name_hist=${safe_selector_name_hist//,/}
        safe_selector_name_hist=${safe_selector_name_hist//=/}
        safe_selector_name_hist=${safe_selector_name_hist//:/}
        selection_file="${AGG_RESULTS_DIR}/${dgp}/selection_history/${safe_selector_name_hist}_SelectionHistory.csv" # Use safe name convention if AggregateResults was fixed
        weight_file="${AGG_RESULTS_DIR}/${dgp}/weight_history/${safe_selector_name_hist}_WeightHistory.csv" # Use safe name convention if AggregateResults was fixed

        # If AggregateResults wasn't fixed, use the original selector name here:
        # selection_file="${AGG_RESULTS_DIR}/${dgp}/selection_history/${selector}_SelectionHistory.csv"
        # weight_file="${AGG_RESULTS_DIR}/${dgp}/weight_history/${selector}_WeightHistory.csv"

        if [ -f "$initial_file" ] && [ -f "$selection_file" ] && [ -f "$weight_file" ]; then
             echo "  Processing Heatmaps for: ${dgp} / ${selector}"
             for seed in "${seeds_to_plot_individually[@]}"; do
                 echo "    Generating Heatmap Plot - Seed: ${seed}"
                 # Pass the ORIGINAL selector name to the python script
                 python "${CODE_DIR}/utils/Auxiliary/PlotWeightHeatmap.py" \
                     --dgp_name "${dgp}" \
                     --selector "${selector}" \
                     --seed "${seed}" \
                     --output_dir "${VIS_OUTPUT_DIR}"
             done
        else
            echo "  Skipping Heatmaps: ${dgp} / ${selector} (Required history files not found)"
            # Optional: Print which file was missing
            # [ ! -f "$initial_file" ] && echo "    Missing: $initial_file"
            # [ ! -f "$selection_file" ] && echo "    Missing: $selection_file"
            # [ ! -f "$weight_file" ] && echo "    Missing: $weight_file"
        fi
    done
done
echo "Weight Heatmap plots generated."


echo ""
echo "--- Visualization Generation Complete ---"