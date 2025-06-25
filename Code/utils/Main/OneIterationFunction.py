# Summary: Runs one full iteration of the active learning process.
# Input: A dictionary SimulationConfigInput with the following keys and values:
#   DataFileInput: A string that indicates the name of the DataFrame in the Data folder.
#   Seed: Seed for reproducability.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially unlabelled and later added to the training set.
#   SelectorType: Selector mechanism in the active learning framework.
#   ModelType: Predictive model.
# Output: A dictionary SimulationResults with the following keys and values:
#   ErrorVec: Vector of errors at each iteration of the learning process.
#   SelectionHistory: Vector of recommended index for query at each iteration of the learning process.
#   SimulationParameters: Parameters used in the simulation.
#   ElapsedTime: Time for the entire learning process.


### Import packages ###
import time
import numpy as np
import math as math
import pandas as pd
import random as random

### Import functions ###
from utils.Auxiliary import LoadData
from utils.Main.LearningProcedure import LearningProcedure 
from utils.Main.TrainTestCandidateSplit import TrainTestCandidateSplit 


### Function ###
def OneIterationFunction(SimulationConfigInput):
    
    ### Set Up ###
    StartTime = time.time()
    random.seed(SimulationConfigInput["Seed"])
    np.random.seed(SimulationConfigInput["Seed"])

    ### Load Data ###
    df = LoadData(SimulationConfigInput["DataFileInput"])

    ### Train Test Candidate Split ###
    df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, 
                                                              SimulationConfigInput["TestProportion"], 
                                                              SimulationConfigInput["CandidateProportion"])

    ### Update SimulationConfig Arguments ###
    SimulationConfigInput['df_Train'] = df_Train
    SimulationConfigInput["df_Test"] = df_Test
    SimulationConfigInput["df_Candidate"] = df_Candidate
    
    ### Learning Process ###
    LearningProcedureOutput = LearningProcedure(SimulationConfigInputUpdated = SimulationConfigInput)
    
    ### Return Simulation Parameters ###
    SimulationParameters = {"DataFileInput" : str(SimulationConfigInput["DataFileInput"]),
                            "Seed" : str(SimulationConfigInput["Seed"]),
                            "TestProportion" : str(SimulationConfigInput["TestProportion"]),
                            "CandidateProportion" : str(SimulationConfigInput["CandidateProportion"]),
                            "SelectorType" :  str(SimulationConfigInput["SelectorType"]),
                            "ModelType" :  str(SimulationConfigInput["ModelType"])
                            }

    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(LearningProcedureOutput["ErrorVec"], columns =["Error"]),
                         "SelectionHistory" : LearningProcedureOutput["SelectedObservationHistory"],
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}

    return SimulationResults