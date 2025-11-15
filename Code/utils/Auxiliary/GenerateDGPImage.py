### Packages ###
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

### Directory ###
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
except NameError:
    code_dir = os.path.abspath(os.path.join(os.getcwd(), 'Code')) # Assumes running from project root

if code_dir not in sys.path:
    print(f"Adding {code_dir} to path")
    sys.path.append(code_dir)

### Import DGP ###
try:
    from utils.Auxiliary.PreprocessData import generate_two_regime_data, generate_three_regime_data
except ImportError as e:
    print(f"Error importing generator functions: {e}")
    print("Please ensure the script is run from a location where it can find the 'Code/utils' directory,")
    print("or adjust the path addition logic above.")
    sys.exit(1)


### White backgrounds (on-screen and saved) ###
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

### Noiseless reference functions ###
def noiseless_two_regime(x):
    y = np.zeros_like(x)
    m1 = x < 0.5
    y[m1] = np.sin(x[m1] * 10 * np.pi)
    y[~m1] = 2 * x[~m1] - 1
    return y

def noiseless_three_regime(x):
    y = np.zeros_like(x)
    m1 = x < 0.4
    m2 = (x >= 0.4) & (x < 0.7)
    m3 = x >= 0.7
    y[m1] = np.sin(x[m1] * 8 * np.pi)
    y[m2] = 3 * x[m2] - 1.5
    y[m3] = 2 * np.cos(x[m3] * 6 * np.pi)
    return y


### Plot helpers (square, white) with dividing lines (no labels) ###
def plot_two_regime(df, save_path=None):
    x = df["X1"].values
    y = df["Y"].values
    m1 = x < 0.5
    m2 = ~m1

    plt.figure(figsize=(6, 6))
    plt.scatter(x[m1], y[m1], s=16, alpha=0.6, label="Regime 1: Sine")
    plt.scatter(x[m2], y[m2], s=16, alpha=0.6, label="Regime 2: Linear")

    # Noiseless guide
    grid = np.linspace(0, 1, 400)
    plt.plot(grid, noiseless_two_regime(grid), color='black', linewidth=2, label="Underlying function") 

    # Dividing lines at regime and trap bounds (no text labels)
    for xpos in [0.5, 0.8, 0.9]:
        plt.axvline(xpos, color='dimgray', linestyle="--", linewidth=1) 

    # Region annotations
    plt.text(0.18, 1.6, "Exploration", fontsize=10)
    plt.text(0.63, -1.6, "Investigation", fontsize=10)
    plt.text(0.805, 1.8, "High-noise trap", fontsize=9)

    plt.title("Two Regime DGP")
    plt.xlabel("X1"); plt.ylabel("Y")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6) 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        # print(f"Saved plot to: {save_path}")
    plt.show()

def plot_three_regime(df, save_path=None):
    x = df["X1"].values
    y = df["Y"].values
    m1 = x < 0.4
    m2 = (x >= 0.4) & (x < 0.7)
    m3 = x >= 0.7

    plt.figure(figsize=(6, 6))
    plt.scatter(x[m1], y[m1], s=16, alpha=0.6, label="Regime 1: Sine")
    plt.scatter(x[m2], y[m2], s=16, alpha=0.6, label="Regime 2: Linear")
    plt.scatter(x[m3], y[m3], s=16, alpha=0.6, label="Regime 3: Cosine")

    # Noiseless guide
    grid = np.linspace(0, 1, 500)
    plt.plot(grid, noiseless_three_regime(grid), color='black', linewidth=2, label="Underlying function") # Changed color

    # Dividing lines at regime and trap bounds (no text labels)
    for xpos in [0.4, 0.7, 0.6, 0.65]:
        plt.axvline(xpos, color='dimgray', linestyle="--", linewidth=1)

    # Region annotations
    plt.text(0.12, 2.2, "Exploration", fontsize=10)
    plt.text(0.47, -2.4, "Investigation", fontsize=10)
    plt.text(0.75, 2.2, "Exploration", fontsize=10)
    plt.text(0.602, 2.6, "High-noise trap", fontsize=9)

    plt.title("Three Regime DGP")
    plt.xlabel("X1"); plt.ylabel("Y")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        # print(f"Saved plot to: {save_path}")
    plt.show()


# --- Run & save ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save DGP visualization plots.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory to save the plot images (e.g., Results/images/manuscript)')
    args = parser.parse_args()    
    SAVE_DIR = Path(args.output_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating DGP plots...")

    df_two = generate_two_regime_data(n_samples=1200, seed=42)
    plot_two_regime(df_two, SAVE_DIR / "dgp_two_regime.png")

    df_three = generate_three_regime_data(n_samples=1600, seed=123)
    plot_three_regime(df_three, SAVE_DIR / "dgp_three_regime.png")

    print(f"Finished generating plots in: {SAVE_DIR.resolve()}")