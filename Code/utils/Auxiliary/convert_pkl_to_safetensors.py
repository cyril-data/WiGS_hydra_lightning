import pandas as pd
import pickle
import os
import argparse
from safetensors.torch import save_file
import torch


def convert_pkl_to_safetensors(pkl_filename):
    """
    Convert a .pkl file (containing a pandas DataFrame) to a .safetensors file.

    Args:
        pkl_filename (str): Name of the .pkl file (including path if not in current directory).
    """
    # Get the current working directory
    cwd = os.getcwd()

    # Define directories to search for the file
    directories = [cwd]

    for directory in directories:
        filepath = os.path.join(directory, pkl_filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

                    if not isinstance(data, pd.DataFrame):
                        raise RuntimeError("The file does not contain a pandas DataFrame.")

                    # Convert DataFrame to a dictionary of tensors
                    # Example: Assume the DataFrame has features and labels
                    # Adjust this part based on your DataFrame structure

                    features = data.iloc[:, 1:].to_numpy().copy()
                    target = data.iloc[:, 0].to_numpy().copy()

                    tensors = {
                        "features": torch.tensor(features, dtype=torch.float32),
                        "targets": torch.tensor(target, dtype=torch.float32),
                    }

                    # Save to .safetensors
                    safetensors_path = os.path.join(
                        directory, pkl_filename.replace(".pkl", ".safetensors")
                    )
                    save_file(tensors, safetensors_path)

                    print(f"✅ File successfully converted to Safetensors: {safetensors_path}")
                    return

            except Exception as e:
                print(f"⚠️ Error processing file {filepath}: {e}")
                continue

    raise FileNotFoundError(
        f"❌ File '{pkl_filename}' not found in any of the specified directories."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .pkl file to .safetensors format.")
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the .pkl file (including path if necessary) to convert.",
    )
    args = parser.parse_args()
    convert_pkl_to_safetensors(args.filename)
