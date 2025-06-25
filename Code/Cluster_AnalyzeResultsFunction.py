### PACKAGES ###
import os
import itertools
import argparse
import numpy as np
import math as math
import pandas as pd 
import random as random
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from utils.Auxiliary import *

### INPUT ###
### GET DIRECTORY ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results/OptimalThreshold")

### PARSER ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, default="-1", help="Job name.")
parser.add_argument("--Data", type=str, default="-1", help="Data type.")
parser.add_argument("--RashomonThreshold", type=float, default="-1.0", help="MAXIMUM Rashomon threshold epislon for TreeFarms.")
parser.add_argument("--Output", type=str, default="-1", help="Output.")
args = parser.parse_args()


### INPUT ###
DataType = args.Data
BaseDirectory = "/Users/simondn/Documents/RashomonActiveLearning/Results/"
PassiveLearningRF = LoadAnalyzedData(DataType, BaseDirectory, "RandomForestClassification", "RFA0")
RandomForestResults = LoadAnalyzedData(DataType, BaseDirectory, "RandomForestClassification", "PLA0")
AnalyzedDataUNREALDUREAL = LoadAnalyzedData(DataType, BaseDirectory, "TreeFarms", argsRashomonThreshold.)

### SHAPE ###
ShapeTable = {"DUREAL":[AnalyzedDataUNREALDUREAL["Error_DUREAL"].shape[0]],
              "UNREAL": [AnalyzedDataUNREALDUREAL["Error_UNREAL"].shape[0]]}
ShapeTable = pd.DataFrame(ShapeTable)
ShapeTable

### RUN TIME ###
TimeTable = {"DUREAL Mean":[str(round(np.mean(AnalyzedDataUNREALDUREAL["Time_DUREAL"])/60,2))],
              "UNREAL Mean": [str(round(np.mean(AnalyzedDataUNREALDUREAL["Time_UNREAL"])/60,2))],
                "DUREAL max":[str(round(np.max(AnalyzedDataUNREALDUREAL["Time_DUREAL"])/60,2))],
              "UNREAL max": [str(round(np.max(AnalyzedDataUNREALDUREAL["Time_UNREAL"])/60,2))]
                         }
TimeTable = pd.DataFrame(TimeTable)
# TimeTable.index = range(10,30,5)
TimeTable

### ERROR VECTOR ###

# Set Up #
PlotSubtitle = f"Data: {DataType} | Iterations: {AnalyzedDataUNREALDUREAL['Error_DUREAL'].shape[0]}"
colors = {
    "PassiveLearning": "black",
    "RandomForest": "green",
    "DUREAL": "orange",
    "UNREAL": "blue"
}

linestyles = {
    "PassiveLearning": "solid",
    "RandomForest": "solid",
    "DUREAL": "solid",
    "UNREAL": "solid"
}

LegendMapping = {
    "DUREAL0": "DUREAL (ε = 0.xxx)",
    "UNREAL0": "UNREAL (ε = 0.xxx)",
}

# markerstyles = {
#     "PassiveLearning": "^",
#     "RandomForest": "^",
#     "DUREAL": "^",
#     "UNREAL": "^"
# }

# Figure #
MeanPlot = MeanVariancePlot(RelativeError = None,
                 PassiveLearning = PassiveLearningRF["Error"],
                 RandomForest = RandomForestResults["Error"],
                #  DUREAL = AnalyzedDataUNREALDUREAL["Error_DUREAL"],
                #  UNREAL = AnalyzedDataUNREALDUREAL["Error_UNREAL"],
                 Colors = colors,
                LegendMapping=LegendMapping,
                 Linestyles=linestyles,
                #  Markerstyles = markerstyles,
                # xlim = [20,25],
                Y_Label = "F1 Score",
                 Subtitle = PlotSubtitle,
                 TransparencyVal = 0.00,
                 VarInput = False,
                #  FigSize = (10,10),
                 CriticalValue = 1.96)