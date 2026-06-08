### Libraries ###
import os
import pickle
import pandas as pd
import numpy as np
from safetensors import safe_open


def read_safetensor(DataFileInput, row_indices=None, all_X_y=True):
    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))

    directories = [cwd, ParentDirectory]

    ### Get Data ###
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".safetensors")

            with safe_open(filepath, framework="pd") as f:
                tensor_slice = f.get_slice("features")
                tensor_features = tensor_slice[:, :]

            with safe_open(filepath, framework="pd") as f:
                tensor_slice = f.get_slice("targets")
                tensor_targets = tensor_slice[:, :]

            print("tensor_features", tensor_features.shape)
            print(
                "tensor_targets",
                tensor_targets.shape,
                np.expand_dims(tensor_targets, axis=1).shape,
            )

            return np.hstack([np.expand_dims(tensor_targets, axis=1), tensor_features])

        except FileNotFoundError:
            continue
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")


def LoadData(DataFileInput):
    """
    Loads the pre-processed data into the simulation script.
    Args:
        DataFileInput: A string that the name of the DataFrame in the Data/processed folder
    Returns:
        data: The data (not yet split into the training, test, and candidate sets) to be used in the active learning process.

    """

    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]

    ### Get Data ###
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")
            with open(filepath, "rb") as file:
                data = pickle.load(file).dropna()
            return data
        except FileNotFoundError:
            continue
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")


def get_target_columns(DataFileInput, column_indices):
    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]

    ### Get Columns ###
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")
            with open(filepath, "rb") as file:
                # Charge le DataFrame complet
                data = pickle.load(file)

                # Vérifie que c'est un DataFrame pandas
                if not isinstance(data, pd.DataFrame):
                    raise RuntimeError("Le fichier ne contient pas un DataFrame pandas.")

                # Sélectionne uniquement les colonnes aux indices spécifiés
                selected_columns = data.iloc[:, column_indices]

                # Supprime le DataFrame original pour libérer la mémoire
                del data

                return selected_columns

        except FileNotFoundError:
            continue
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")


def get_target_columns_except_first(DataFileInput):
    """
    Load a DataFrame from a .pkl file and return all columns except the first one.

    Args:
        DataFileInput (str): Name of the file (without extension) to load from the "Data/processed" directory.

    Returns:
        pd.DataFrame: DataFrame containing all columns except the first one.

    Raises:
        FileNotFoundError: If the file is not found in any of the specified directories.
        RuntimeError: If an error occurs while loading the file or if the file is not a pandas DataFrame.
    """

    # Define directories to search for the file
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]

    # Search for the file in each directory
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")

            # Load the DataFrame from the .pkl file
            with open(filepath, "rb") as file:
                data = pickle.load(file)

                # Check if the loaded object is a pandas DataFrame
                if not isinstance(data, pd.DataFrame):
                    raise RuntimeError("The file does not contain a pandas DataFrame.")

                # Select all columns except the first one (i.e., columns from index 1 onwards)
                selected_columns = data.iloc[:, 1:]

                # Free memory by deleting the original DataFrame
                del data

                return selected_columns

        except FileNotFoundError:
            # Continue to the next directory if the file is not found
            continue
        except Exception as e:
            # Raise an error if something else goes wrong
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    # If the file is not found in any directory, raise an error
    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")


def get_target_rows(DataFileInput, row_indices):
    """
    Load a DataFrame from a .pkl file and return only the specified rows by their indices.

    Args:
        DataFileInput (str): Name of the file (without extension) to load from the "Data/processed" directory.
        row_indices (list): List of row indices to select (e.g., [0, 13, 9, 45, 33]).

    Returns:
        pd.DataFrame: DataFrame containing only the selected rows.

    Raises:
        FileNotFoundError: If the file is not found in any of the specified directories.
        RuntimeError: If an error occurs while loading the file or if the file is not a pandas DataFrame.
    """

    # Define directories to search for the file
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]

    # Search for the file in each directory
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")

            # Load the DataFrame from the .pkl file
            with open(filepath, "rb") as file:
                data = pickle.load(file)

                # Check if the loaded object is a pandas DataFrame
                if not isinstance(data, pd.DataFrame):
                    raise RuntimeError("The file does not contain a pandas DataFrame.")

                # Select only the specified rows
                selected_rows = data.iloc[row_indices, :]

                # Free memory by deleting the original DataFrame
                del data

                return selected_rows

        except FileNotFoundError:
            # Continue to the next directory if the file is not found
            continue
        except Exception as e:
            # Raise an error if something else goes wrong
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    # If the file is not found in any directory, raise an error
    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")


def get_target_rows_col(DataFileInput, row_indices, column_indices):
    """
    Load a DataFrame from a .pkl file and return only the specified rows by their indices.

    Args:
        DataFileInput (str): Name of the file (without extension) to load from the "Data/processed" directory.
        row_indices (list): List of row indices to select (e.g., [0, 13, 9, 45, 33]).

    Returns:
        pd.DataFrame: DataFrame containing only the selected rows.

    Raises:
        FileNotFoundError: If the file is not found in any of the specified directories.
        RuntimeError: If an error occurs while loading the file or if the file is not a pandas DataFrame.
    """

    # Define directories to search for the file
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]

    # Search for the file in each directory
    for directory in directories:
        try:
            filepath = os.path.join(directory, "Data", "processed", DataFileInput + ".pkl")

            # Load the DataFrame from the .pkl file
            with open(filepath, "rb") as file:
                data = pickle.load(file)

                # Check if the loaded object is a pandas DataFrame
                if not isinstance(data, pd.DataFrame):
                    raise RuntimeError("The file does not contain a pandas DataFrame.")

                # Select only the specified rows
                selected_rows = data.iloc[row_indices, column_indices]

                # Free memory by deleting the original DataFrame
                del data

                return selected_rows

        except FileNotFoundError:
            # Continue to the next directory if the file is not found
            continue
        except Exception as e:
            # Raise an error if something else goes wrong
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    # If the file is not found in any directory, raise an error
    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")
