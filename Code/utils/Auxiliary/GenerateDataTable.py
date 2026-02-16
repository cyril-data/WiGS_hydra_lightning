import pandas as pd
import os

# === CONFIGURATION ===
# Path to your processed datasets
# Assuming script is run from project root 'WeightedGreedySampling'
DATA_DIR = "Data/processed/" 
OUTPUT_DIR = "Results/tables/"
OUTPUT_FILENAME = "DatasetTable.tex"

# Map KEYS (Actual Filenames without extension) to (Display Name, Source)
# The KEY on the left must match the filename in 'ls' exactly!
META_INFO = {
    "mpg":                   ("AutoMPG",                  "UCI ML Repository"),
    "beer":                  ("Beer",                     "Kaggle"),
    "bodyfat":               ("Body Fat",                 "Kaggle"),
    "burbidge_correct":      ("Burbidge - Correct",       "\\citet{Burbridge}"),
    "burbidge_misspecified": ("Burbidge - Misspecified",  "\\citet{Burbridge}"),
    "burbidge_low_noise":    ("Burbidge - Low Noise",     "\\citet{Burbridge}"),
    "concrete_4":            ("Concrete",                 "UCI ML Repository"),
    "concrete_cs":           ("Concrete - CS",            "UCI ML Repository"),
    "concrete_flow":         ("Concrete - Flow",          "UCI ML Repository"),
    "concrete_slump":        ("Concrete - Slump",         "UCI ML Repository"),
    "cps_wage":              ("CPS",                      "CMU Stat Lib"),
    "housing":               ("Housing",                  "UCI ML Repository"),
    "no2":                   ("NO2",                      "CMU Stat Lib"),
    "pm10":                  ("PM10",                     "CMU Stat Lib"),
    "qsar":                  ("QSAR",                     "UCI ML Repository"),
    "wine_red":              ("Wine - Red",               "UCI ML Repository"),
    "wine_white":            ("Wine - White",             "UCI ML Repository"),
    "yacht":                 ("Yacht",                    "UCI ML Repository"),
}

def get_feature_count(df):
    """
    Returns number of features. 
    Assumes the last column is the Target, so Features = Total Cols - 1.
    """
    return df.shape[1] - 1

def generate_latex_table():
    print(f"Scanning {DATA_DIR} for datasets...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    rows = []
    
    # Iterate through the dictionary to preserve order
    for i, (file_key, (display_name, source)) in enumerate(META_INFO.items(), 1):
        
        # Construct path (checks for .csv, then .pkl)
        csv_path = os.path.join(DATA_DIR, f"{file_key}.csv")
        pkl_path = os.path.join(DATA_DIR, f"{file_key}.pkl")
        
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            elif os.path.exists(pkl_path):
                df = pd.read_pickle(pkl_path)
            else:
                print(f"Warning: Could not find file for key '{file_key}' (Checked {pkl_path})")
                rows.append([i, display_name, source, "N/A", "N/A"])
                continue

            # Calculate stats
            n_samples = df.shape[0]
            n_features = get_feature_count(df)
            
            rows.append([i, display_name, source, n_samples, n_features])

        except Exception as e:
            print(f"Error reading {file_key}: {e}")

    # === Generate LaTeX String ===
    latex_content = r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{lllll}
        \toprule
        \textbf{no.} & \textbf{Dataset} & \textbf{Source} & \textbf{Size} & \textbf{Features} \\ \midrule
"""
    
    for row in rows:
        # F-string formatting for alignment
        line = f"        {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n"
        latex_content += line
        
    latex_content += r"""        \bottomrule
    \end{tabular}
    \caption{Benchmark datasets used in the experiments.}
    \label{tab:DatasetsTable}
\end{table}
"""

    # === Write to File ===
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        f.write(latex_content)
        
    print(f"Table saved to: {output_path}")

if __name__ == "__main__":
    generate_latex_table()