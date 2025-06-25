### Import Packages ###
import os
import argparse
import numpy as np
import math as math
import pandas as pd
from scipy import stats
import random as random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from treeFarms.treefarms.model.treefarms import TREEFARMS

### IMPORT FUNCTIONS ###
from utils.Auxiliary import *

### GET DIRECTORY ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results/OptimalThreshold")

### PARSER ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, default="-1", help="Job name.")
parser.add_argument("--Seed", type=int, default=-1, help="Seed.")
parser.add_argument("--Data", type=str, default="-1", help="Data type.")
parser.add_argument("--TestProportion", type=float, default="-1.0", help="Percent for validation set.")
# parser.add_argument("--CandidateProportion", type=float, default="-1.0", help="Percent for candidate datset.")
parser.add_argument("--regularization", type=float, default="-1.0", help="Regularization for TreeFarms.")
parser.add_argument("--RashomonThresholdType", type=str, default="-1", help="Adder or multiplier.")
parser.add_argument("--RashomonThreshold", type=float, default="-1.0", help="MAXIMUM Rashomon threshold epislon for TreeFarms.")
parser.add_argument("--Output", type=str, default="-1", help="Output.")
args = parser.parse_args()


### SET UP ###

# Input #
DataFile = args.Data
rashomon_bound_adder = float(args.RashomonThreshold)
regularization = float(args.regularization)
TestProportion = float(args.TestProportion)
# CandidateProportion = float(args.CandidateProportion)
Seed = args.Seed

# Load Data #
df = LoadData(DataFile)
random.seed(Seed)
np.random.seed(Seed)

# Train Test Candidate Split #
# from utils.Main import TrainTestCandidateSplit
# df_Train, df_Test, df_Candidate = TrainTestCandidateSplit(df, TestProportion, CandidateProportion)
# Train Test Split #
X_Train, X_Test, y_Train, y_Test = train_test_split(df.loc[:, df.columns != "Y"], df["Y"], test_size=TestProportion)
df_Train = X_Train.copy()
df_Train.insert(0, 'Y', y_Train)
df_Test = X_Test.copy()
df_Test.insert(0, 'Y', y_Test)


### TRAIN TREEFARMS ###
# TreeFarms #
config = {"regularization": regularization, "rashomon_bound_adder": rashomon_bound_adder}
TreeFarmsModel = TREEFARMS(config)
TreeFarmsModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
TreeCount = TreeFarmsModel.get_tree_count()

# Duplicate and Unique #
# PredictionArray_Duplicate = pd.DataFrame(np.array([TreeFarmsModel[i].predict(df_Train.loc[:, df_Train.columns != "Y"]) for i in range(TreeCount)]))
# PredictionArray_Unique = pd.DataFrame(PredictionArray_Duplicate).drop_duplicates(keep='first', ignore_index=False)
# TrueValues = df_Train["Y"].to_numpy()
# PredictionArray = PredictionArray_Unique


### TRAINING ACCURACY ###
# Training Accuracy #
# TreeClassificationAccuracy = PredictionArray.eq(TrueValues, axis=1).mean(axis=1)
# BestAccuracy = float(np.max(TreeClassificationAccuracy))
TrainingAccuracy = [1-TreeFarmsModel[i].error(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"]) for i in range(TreeCount)]


# Threshold Values #
# EpsilonVec = BestAccuracy - TreeClassificationAccuracy
EpsilonVec = np.max(TrainingAccuracy) - TrainingAccuracy
# MinEpsilon = float(np.min(EpsilonVec))
# MaxEpsilon = float(np.max(EpsilonVec))
# ThresholdValues = np.arange(MinEpsilon, MaxEpsilon + 0.000001, 0.000001)
ThresholdValues = np.arange(0, 1.25*rashomon_bound_adder, 0.000001)

### TEST ACCURACY ###
# Set Up #
ModelIndicesVec = []
Epsilon_F1Score = []
Epsilon_ClassAccuracy = []
ModelIndicesOld = []
F1ScoreOld = None
ClassAccuracyOld = None

# Search #
for Threshold in ThresholdValues:

    # Filter Models Based on Threshold
    ModelIndices = EpsilonVec[EpsilonVec <= Threshold].index.tolist()

    # Only recalculate F1 and Class Accuracy if new models were added
    if ModelIndices == ModelIndicesOld:
        # Use stored values
        F1Score = F1ScoreOld
        ClassAccuracy = ClassAccuracyOld
    else: 
        # Test Set Predictions
        Test_Predictions = pd.DataFrame(
            np.array([TreeFarmsModel[i].predict(df_Test.loc[:, df_Test.columns != "Y"]) for i in ModelIndices])
        )
        Test_Predictions.columns = df_Test.index.astype(str)

        # Compute Ensemble Prediction (Mode)
        mode_result = stats.mode(Test_Predictions, axis=0, keepdims=True)
        EnsemblePrediction = pd.Series(mode_result.mode.flatten())
        EnsemblePrediction.index = df_Test.index

        # Compute Metrics
        F1Score = float(f1_score(df_Test["Y"], EnsemblePrediction, average='micro'))
        ClassAccuracy = float(np.mean(EnsemblePrediction == df_Test["Y"]))

        # Store Old ModelIndices
        ModelIndicesOld = ModelIndices.copy()
        F1ScoreOld = F1Score
        ClassAccuracyOld = ClassAccuracy

    # Append Metrics
    ModelIndicesVec.append(ModelIndices)
    Epsilon_F1Score.append(F1Score)
    Epsilon_ClassAccuracy.append(ClassAccuracy)

### OUTPUT ###
SimulationResults = {
    "ModelIndicesVec" : ModelIndicesVec,
    "ThresholdValues" : ThresholdValues,
    "Epsilon_F1Score" : Epsilon_F1Score,
    "Epsilon_ClassAccuracy" : Epsilon_ClassAccuracy}

### Save Simulation Results ###
with open(os.path.join(SaveDirectory, str(args.Data), "Raw", str(args.Output)), 'wb') as f:
    pickle.dump(SimulationResults, f)