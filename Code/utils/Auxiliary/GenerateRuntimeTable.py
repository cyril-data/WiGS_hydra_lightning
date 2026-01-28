### Libraries ###
import os
import pandas as pd
import numpy as np

### CONFIGURATION ###
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", "simulation_results", "aggregated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Results", "tables")
OUTPUT_FILENAME = "RuntimeTable.tex"
os.makedirs(OUTPUT_DIR, exist_ok=True)

### COLUMN GROUPING CONFIGURATION ###
COLUMN_GROUPS = [
    {
        "group_name": "Baselines",
        "columns": [
            ("Passive Learning", "Rand."),
            ("GSx", "GSx"),
            ("GSy", "GSy"),
            ("iGS", "iGS")
        ]
    },
    {
        "group_name": "Advanced Baselines",
        "columns": [
            ("QBC", "QBC"),
            ("Uncertainty Sampling", "Uncert."),
            ("EMCM", "EMCM"),
            ("Information Density", "InfoDen")
        ]
    },
    {
        "group_name": "WiGS (Static/Decay)",
        "columns": [
            ("WiGS (Static w_x=0.25)", "0.25"),
            ("WiGS (Static w_x=0.75)", "0.75"),
            ("WiGS (Time-Decay, Linear)", "Lin."),
            ("WiGS (Time-Decay, Exponential)", "Exp.")
        ]
    },
    {
        "group_name": "WiGS (Adaptive)",
        "columns": [
            ("WiGS (MAB-UCB1, c=2.0)", "MAB(2)"),
            ("WiGS (SAC)", "SAC")
        ]
    }
]

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
                
                # Format Dataset Name for LaTeX
                ds_name = dataset.replace("concrete", "Conc.").replace("burbidge", "Burb.").replace("wine", "Wine").title().replace("_", " ")
                # Escape underscores for LaTeX
                ds_name = ds_name.replace("_", "\\_")
                
                row = {"Dataset": ds_name}
                
                # Extract Median
                for col in all_target_cols:
                    if col in df.columns:
                        row[col] = df[col].median()
                    else:
                        row[col] = float('nan') 
                
                results.append(row)
            except Exception as e:
                print(f"Skipping {dataset}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Median Row
    if not df.empty:
        avg_row = df.median(numeric_only=True)
        avg_row["Dataset"] = "\\textbf{MEDIAN}"
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    ### GENERATE LATEX ###
    latex_content = r"\begin{table*}[htbp]" + "\n" # Use table* for wide table spanning both columns
    latex_content += r"    \centering" + "\n"
    latex_content += r"    \scriptsize" + "\n"
    latex_content += r"    \setlength{\tabcolsep}{2pt}" + "\n" # Tighten space
    
    # Build alignment string
    total_data_cols = sum(len(g["columns"]) for g in COLUMN_GROUPS)
    latex_content += f"    \\begin{{tabular}}{{l {'r' * total_data_cols}}}" + "\n"
    latex_content += r"        \toprule" + "\n"
    
    ## HEADER ROW 1: GROUPS ##
    header1 = "        " 
    cmidrules = ""
    current_col_idx = 2 # Start at 2 because col 1 is Dataset
    
    for group in COLUMN_GROUPS:
        group_name = group["group_name"]
        num_sub_cols = len(group["columns"])
        
        header1 += f" & \\multicolumn{{{num_sub_cols}}}{{c}}{{\\textbf{{{group_name}}}}}"
        
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
        
        # Bold the MEDIAN row logic is handled by the name being bolded already
        latex_content += f"        {ds_name}"
            
        # Iterate through groups to maintain order
        for group in COLUMN_GROUPS:
            for col_key, _ in group["columns"]:
                val = row[col_key]
                
                if pd.isna(val):
                    latex_content += " & -"
                else:
                    if "MEDIAN" in ds_name:
                        latex_content += f" & \\textbf{{{val:.1f}}}" # 1 decimal for space saving
                    else:
                        latex_content += f" & {val:.1f}"
                        
        latex_content += r" \\" + "\n"
        
        # Add a midrule before the final MEDIAN row
        if _ == len(df) - 2:
             latex_content += r"        \midrule" + "\n"

    latex_content += r"        \bottomrule" + "\n"
    latex_content += r"    \end{tabular}" + "\n"
    latex_content += r"    \caption{Median runtime (seconds) across simulation seeds. `Uncert.' denotes Uncertainty Sampling, `InfoDen' denotes Information Density.}" + "\n"
    latex_content += r"    \label{tab:RuntimeComparison}" + "\n"
    latex_content += r"\end{table*}" + "\n"

    ### Write to File ###
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        f.write(latex_content)
        
    print(f"\nLaTeX Table saved to: {output_path}")

if __name__ == "__main__":
    main()