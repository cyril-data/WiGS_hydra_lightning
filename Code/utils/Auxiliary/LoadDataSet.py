# Summary: Loads the pre-processed data into the simulation script.
# Input: 
#   DataFileInput: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder
# Output: 
#   data: The data (not yet split into the training, test, and candidate sets) to be used in the active learning process.

### Libraries ###
import os
import pickle
import pandas as pd

def LoadData(DataFileInput):
    
    ### Directory ###
    cwd = os.getcwd()
    ParentDirectory = os.path.abspath(os.path.join(cwd, "../"))
    ScratchParentDirectory = os.path.abspath(os.path.join(cwd, "../../"))
    directories = [cwd, ParentDirectory, ScratchParentDirectory]  # Cluster first, then local, then within Scratch

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