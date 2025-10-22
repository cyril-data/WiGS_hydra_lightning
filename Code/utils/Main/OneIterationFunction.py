### Import packages ###
import time
import numpy as np
import pandas as pd
import random as random

### Import functions ###
from utils.Auxiliary import LoadData
from utils.Main.LearningProcedure import LearningProcedure 
from utils.Main.TrainCandidateSplit import TrainCandidateSplit 

### Function ###
def OneIterationFunction(SimulationConfigInput):
    """
    Executes a single, complete simulation iteration for a given configuration.

    Args:
        SimulationConfigInput (dict): A dictionary containing all parameters needed to run the simulation. 

    Returns:
        dict: A dictionary containing the results of the simulation run, with the following keys:
            - ErrorVecs (pd.DataFrame): The history of performance metrics over the course of the learning procedure.
            - SelectionHistory (list): The history of observations selected from the candidate pool.
            - SimulationParameters (dict): The key input parameters used for this simulation run.
            - ElapsedTime (float): The total execution time in seconds for this iteration."""
    
    ### Set Up ###
    StartTime = time.time()
    random.seed(SimulationConfigInput["Seed"])
    np.random.seed(SimulationConfigInput["Seed"])

    ### Load Data ###
    df = LoadData(SimulationConfigInput["DataFileInput"])

    ### Train Candidate Split ###
    df_Train, df_Candidate = TrainCandidateSplit(df, SimulationConfigInput["CandidateProportion"])

    ### Update SimulationConfig Arguments ###
    SimulationConfigInput['df_Train'] = df_Train
    SimulationConfigInput["df_Candidate"] = df_Candidate
    
    ### Learning Process ###
    LearningProcedureOutput = LearningProcedure(SimulationConfigInputUpdated = SimulationConfigInput)
    
    ### Return Simulation Parameters ###
    SimulationParameters = {"DataFileInput" : str(SimulationConfigInput["DataFileInput"]),
                            "Seed" : str(SimulationConfigInput["Seed"]),
                            "CandidateProportion" : str(SimulationConfigInput["CandidateProportion"]),
                            "SelectorType" :  str(SimulationConfigInput["SelectorType"]),
                            "ModelType" :  str(SimulationConfigInput["ModelType"])
                            }

    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    ErrorVecs = pd.DataFrame(LearningProcedureOutput["ErrorVecs"])

    SimulationResults = {"ErrorVecs" : ErrorVecs,
                         "SelectionHistory" : LearningProcedureOutput["SelectedObservationHistory"],
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}

    return SimulationResults