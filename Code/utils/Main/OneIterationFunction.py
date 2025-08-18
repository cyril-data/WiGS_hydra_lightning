### Import packages ###
import time
import numpy as np
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
    ErrorVec = pd.DataFrame(LearningProcedureOutput["ErrorVec"])

    SimulationResults = {"ErrorVec" : ErrorVec,
                         "SelectionHistory" : LearningProcedureOutput["SelectedObservationHistory"],
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}

    return SimulationResults