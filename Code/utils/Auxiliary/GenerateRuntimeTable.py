### Libraries ###
import os
import pandas as pd
import numpy as np

### CONFIGURATION ###
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "simulation_results", "aggregated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Results", "tables")
OUTPUT_FILENAME = "RuntimeTable.tex"
os.makedirs(OUTPUT_DIR, exist_ok=True)

### COLUMN GROUPING CONFIGURATION ###
COLUMN_GROUPS = [
    {
        "group_name": "Baselines",
        "columns": [
            ("Passive Learning", "Random"),
            ("GSx", "GSx"),
            ("GSy", "GSy"),
            ("iGS", "iGS"),
            ("QBC", "QBC") 
        ]
    },
    {
        "group_name": "WiGS (Static)",
        "columns": [
            ("WiGS (Static w_x=0.25)", "0.25"),
            ("WiGS (Static w_x=0.5)", "0.50"),
            ("WiGS (Static w_x=0.75)", "0.75")
        ]
    },
    {
        "group_name": "WiGS (Decay)",
        "columns": [
            ("WiGS (Time-Decay, Linear)", "Linear"),
            ("WiGS (Time-Decay, Exponential)", "Exp.")
        ]
    },
    {
        "group_name": "WiGS (Adaptive)",
        "columns": [
            ("WiGS (MAB-UCB1, c=0.5)", "MAB (0.5)"),
            ("WiGS (MAB-UCB1, c=2.0)", "MAB (2.0)"),
            ("WiGS (SAC)", "SAC")
        ]
    }
]

# Columns to ignore during initial read (standard cleanup)
IGNORE_COLS = ["Simulation", "Unnamed: 0"]

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    datasets = sorted([d for d in os.listdir(RESULTS_DIR) if not d.startswith(".")])
    results = []

    print(f"Generating Grouped Runtime Table for {len(datasets)} datasets...")

    # Flatten the config for easy data extraction
    all_target_cols = []
    for group in COLUMN_GROUPS:
        for csv_col, _ in group["columns"]:
            all_target_cols.append(csv_col)

    for dataset in datasets:
        file_path = os.path.join(RESULTS_DIR, dataset, "ElapsedTime.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Format Dataset Name
                ds_name = dataset.replace("concrete", "Conc.").replace("burbidge", "Burb.").replace("wine", "Wine").title().replace("_", " ")
                row = {"Dataset": ds_name}
                
                # Extract Means
                for col in all_target_cols:
                    if col in df.columns:
                        row[col] = df[col].mean()
                    else:
                        row[col] = float('nan') 
                
                results.append(row)
            except Exception as e:
                print(f"Skipping {dataset}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Average
    if not df.empty:
        avg_row = df.mean(numeric_only=True)
        avg_row["Dataset"] = "AVERAGE"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    ### GENERATE LATEX ###
    latex_content = r"\begin{table}[htbp]" + "\n"
    latex_content += r"    \centering" + "\n"
    latex_content += r"    \scriptsize" + "\n"
    latex_content += r"    \setlength{\tabcolsep}{3pt}" + "\n" # Tighten space between columns
    
    # Build alignment string: l (Dataset) + r (Data)...
    total_data_cols = sum(len(g["columns"]) for g in COLUMN_GROUPS)
    latex_content += f"    \\begin{{tabular}}{{l {'r' * total_data_cols}}}" + "\n"
    latex_content += r"        \toprule" + "\n"
    
    ## HEADER ROW 1: GROUPS ##
    header1 = "        " # Empty cell for 'Dataset'
    cmidrules = ""
    current_col_idx = 2 # Start at 2 because col 1 is Dataset
    
    for group in COLUMN_GROUPS:
        group_name = group["group_name"]
        num_sub_cols = len(group["columns"])
        
        # Add the multicell header
        header1 += f" & \\multicolumn{{{num_sub_cols}}}{{c}}{{\\textbf{{{group_name}}}}}"
        
        # Add the line underneath it (trim left and right to make gaps)
        end_col_idx = current_col_idx + num_sub_cols - 1
        cmidrules += f" \\cmidrule(lr){{{current_col_idx}-{end_col_idx}}}"
        
        current_col_idx += num_sub_cols
        
    latex_content += header1 + r" \\" + "\n"
    latex_content += "        " + cmidrules + "\n"
    
    ## HEADER ROW 2: SUB-HEADERS ##
    header2 = "        \\textbf{Dataset}"
    for group in COLUMN_GROUPS:
        for _, short_label in group["columns"]:
            header2 += f" & {short_label}"
            
    latex_content += header2 + r" \\ \midrule" + "\n"

    # DATA ROWS ---
    for _, row in df.iterrows():
        ds_name = row['Dataset']
        
        # Bold the AVERAGE row
        if ds_name == "AVERAGE":
            latex_content += r"        \midrule" + "\n"
            latex_content += r"        \textbf{AVERAGE}"
        else:
            latex_content += f"        {ds_name}"
            
        # Iterate through groups to maintain order
        for group in COLUMN_GROUPS:
            for col_key, _ in group["columns"]:
                val = row[col_key]
                
                if pd.isna(val):
                    latex_content += " & -"
                else:
                    if ds_name == "AVERAGE":
                        latex_content += f" & \\textbf{{{val:.2f}}}"
                    else:
                        latex_content += f" & {val:.2f}"
                        
        latex_content += r" \\" + "\n"

    latex_content += r"        \bottomrule" + "\n"
    latex_content += r"    \end{tabular}" + "\n"
    latex_content += r"    \caption{Average runtime (seconds) across 100 simulations.}" + "\n"
    latex_content += r"    \label{tab:RuntimeComparison}" + "\n"
    latex_content += r"\end{table}" + "\n"

    ### Write to File ###
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        f.write(latex_content)
        
    print(f"\nLaTeX Table saved to: {output_path}")

if __name__ == "__main__":
    main()