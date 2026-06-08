import pandas as pd
import pickle
import os
import argparse


def convert_pkl_to_parquet_and_hdf5(pkl_filename):
    """
    Convert a .pkl file to .parquet and .h5 (HDF5) formats.

    Args:
        pkl_filename (str): Name of the .pkl file (including path if not in current directory).
    """
    # Get the current working directory
    cwd = os.getcwd()
    print("Current working directory:", cwd)

    # Define directories to search for the file (current, parent, grandparent, and great-grandparent)
    directories = [
        cwd,
        os.path.abspath(os.path.join(cwd, "../")),
        os.path.abspath(os.path.join(cwd, "../../")),
        os.path.abspath(os.path.join(cwd, "../../../")),
    ]

    # Iterate through each directory to find the file
    for directory in directories:
        filepath = os.path.join(directory, pkl_filename)
        if os.path.exists(filepath):
            try:
                # Load the data from the .pkl file
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

                    # Check if the loaded data is a pandas DataFrame
                    if not isinstance(data, pd.DataFrame):
                        raise RuntimeError("The file does not contain a pandas DataFrame.")

                    # Save as Parquet
                    parquet_path = os.path.join(directory, pkl_filename + ".parquet")
                    data.to_parquet(parquet_path, engine="pyarrow")

                    # Save as HDF5
                    hdf5_path = os.path.join(directory, pkl_filename + ".h5")
                    data.to_hdf(hdf5_path, key="data", mode="w", format="table")

                    print(
                        f"✅ File successfully converted to Parquet and HDF5: {parquet_path} and {hdf5_path}"
                    )
                    return

            except Exception as e:
                print(f"⚠️ Error processing file {filepath}: {e}")
                continue

    # If the file is not found in any directory, raise an error
    raise FileNotFoundError(
        f"❌ File '{pkl_filename}' not found in any of the specified directories."
    )


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a .pkl file to .parquet and .h5 (HDF5) formats."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the .pkl file (including path if necessary) to convert.",
    )
    args = parser.parse_args()

    # Call the conversion function
    convert_pkl_to_parquet_and_hdf5(args.filename)
