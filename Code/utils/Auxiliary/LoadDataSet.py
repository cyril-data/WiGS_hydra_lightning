### Libraries ###
import os
import pickle
import pandas as pd

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
            with open(filepath, 'rb') as file:
                data = pickle.load(file).dropna()
            return data
        except FileNotFoundError:
            continue
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")

    raise FileNotFoundError(f"File '{DataFileInput}.pkl' not found in any specified directories.")