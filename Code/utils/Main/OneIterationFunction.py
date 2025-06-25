# Summary: Runs one full iteration of the active learning process.
# Input: A dictionary SimulationConfigInput with the following keys and values:
#   DataFileInput: A string that indicates the name of the DataFrame in the Data folder.
#   Seed: Seed for reproducability.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially unlabelled and later added to the training set.
#   SelectorType: Selector mechanism in the active learning framework.
#   ModelType: Predictive model.
#   UniqueErrorsInput: A binary input indicating whether to prune duplicate trees in TreeFARMS.
#   n_estimators: The number of weak learners used for a random forest.
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThresholdType: A string {"Adder", "Multiplier"} indicating whether the Rashomon threshold is added or multiplied.
#   RashomonThreshold: A float indicating the Rashomon threshold in TreeFARMS.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
#   DiversityWeight: The weight placed on the diversity of the data inputs for the selection mechanism in batch active learning.
#   DensityWeight: The weight placed on the density of the data inputs for the selection mechanism in batch active learning.
#   BatchSize: The number of observations to be queried in batch active learning.
# Output: A dictionary SimulationResults with the following keys and values:
#   ErrorVec: Vector of errors at each iteration of the learning process.
#   EpsilonVec: Vector of Rashomon threshold values at each iteration.
#   TreeCount: A dictionary that contains two keys: {AllModelsInRashomonSet, UniqueModelsInRashomonSet} indicating
#                          the number of trees in the Rashomon set from TreeFARMS and the number of unique classification patterns.
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
from utils.Auxiliary import DiversityMetricsFunction
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

    ### Add Batch Active Learning Metrics ###
    df_Candidate = DiversityMetricsFunction(df_Candidate, df_Train, k=10)
    SimulationConfigInput['auxiliary_data_cols'] = ['DiversityScores', 'DensityScores']

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
                            "ModelType" :  str(SimulationConfigInput["ModelType"]),
                            'UniqueErrorsInput': str(SimulationConfigInput["UniqueErrorsInput"]),
                            'n_estimators': str(SimulationConfigInput["n_estimators"]),
                            'regularization': str(SimulationConfigInput["regularization"]),
                            'RashomonThresholdType': str(SimulationConfigInput["RashomonThresholdType"]),
                            'RashomonThreshold': str(SimulationConfigInput["RashomonThreshold"]),
                            'Type': 'Classification',
                            'DiversityWeight': str(SimulationConfigInput["DiversityWeight"]),
                            'DensityWeight': str(SimulationConfigInput["DensityWeight"]),
                            'BatchSize': str(SimulationConfigInput["BatchSize"])
                            }
    # Add model-specific parameters to SimulationParameters if they exist in the config
    for key in ['hidden_size', 
                'dropout_rate', 
                'epochs', 
                'learning_rate', 
                'batch_size_train',
                'K_BALD_Samples', 
                'kernel_type', 
                'kernel_length_scale', 
                'kernel_nu',
                'optimizer', 
                'n_restarts_optimizer', 
                'max_iter_predict', 
                'RefitFrequency',
                'auto_tune_epsilon']: 
        
        if key in SimulationConfigInput:
            SimulationParameters[key] = str(SimulationConfigInput[key])
    

    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    SimulationResults = {"ErrorVec" : pd.DataFrame(LearningProcedureOutput["ErrorVec"], columns =["Error"]),
                         "EpsilonVec" : pd.DataFrame(LearningProcedureOutput["EpsilonVec"], columns =["Epsilon"]),
                         "RefitDecisionVec" : pd.DataFrame(LearningProcedureOutput["RefitDecisionVec"], columns=["RefitDecision"]), 
                         "TreeCount": LearningProcedureOutput["TreeCount"],
                         "SelectionHistory" : LearningProcedureOutput["SelectedObservationHistory"],
                         "SimulationParameters" : SimulationParameters,
                         "ElapsedTime" : ElapsedTime}


    return SimulationResults