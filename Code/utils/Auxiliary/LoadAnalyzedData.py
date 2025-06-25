### Packages ###
import os
import pickle
import pandas as pd

### Function ###
def LoadAnalyzedData(data_type, base_directory, model_directory, file_prefix):
    """
    Loads all available analyzed data for a specific simulation run based on its file prefix.

    This function is designed to be generic and will attempt to load a standard set of
    result files without needing to know the specifics of the model (e.g., RF vs BNN).

    Args:
        data_type (str): The name of the data set (e.g., "Iris").
        base_directory (str): The root directory where all results are stored.
        model_directory (str): The directory name for the specific model predictor
                               (e.g., "BayesianNeuralNetworkPredictor").
        file_prefix (str): The unique identifier for the simulation run, used as a
                           prefix in filenames (e.g., "_BNN_BALD").

    Returns:
        dict: A dictionary containing all the data found for the given prefix.
              Keys for which no file was found will have a value of None.
    """

    ResultsDirectory = os.path.join(base_directory, data_type, model_directory, "ProcessedResults")

    # Define the standard metrics to look for and their associated subdirectories and filename suffixes.
    # The key is the name we'll use in the output dictionary.
    # The value is a tuple of (Subdirectory, Filename Suffix).
    path_templates = {
        "Error":            ("ErrorVec", f"{file_prefix}__ErrorMatrix.csv"),
        "Time":             ("ElapsedTime", f"{file_prefix}__TimeMatrix.csv"),
        "SelectionHistory": ("SelectionHistory", f"{file_prefix}__SelectionHistory.pkl"),
        "AllTreeCount":     ("TreeCount", f"{file_prefix}__AllTreeCount.csv"),
        "UniqueTreeCount":  ("TreeCount", f"{file_prefix}__UniqueTreeCount.csv"),
        "Epsilon":          ("EpsilonVec", f"{file_prefix}__EpsilonMatrix.csv"),
        "RefitDecision":    ("RefitDecisionVec", f"{file_prefix}__RefitDecisionMatrix.csv"),
    }

    DataDictionary = {}
    for key, (subdir, filename) in path_templates.items():
        FullPath = os.path.join(ResultsDirectory, subdir, filename)
        DataDictionary[key] = None  # Initialize with None

        # Try to load the file if it exists
        if os.path.exists(FullPath):
            try:
                if filename.endswith(".csv"):
                    DataDictionary[key] = pd.read_csv(FullPath)
                elif filename.endswith(".pkl"):
                    with open(FullPath, 'rb') as file:
                        DataDictionary[key] = pickle.load(file)
            except Exception as e:
                print(f"Warning: Could not load file {FullPath}. Error: {e}")
        # If the file doesn't exist, we just leave the value as None.

    # Check if any data was loaded at all
    if all(value is None for value in DataDictionary.values()):
        print(f"Warning: No data files found for prefix '{file_prefix}' in '{model_directory}'.")

    return DataDictionary