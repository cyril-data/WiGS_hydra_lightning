### Import Packages ###
import ast
import argparse
import json
import numpy as np
import math as math
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

### Get Directory ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results")

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, default="-1", help="Job name.")
parser.add_argument("--Seed", type=int, default=-1, help="Seed.")
parser.add_argument("--Data", type=str, default="-1", help="Data type.")
parser.add_argument("--TestProportion", type=float, default="-1.0", help="Percent for validation set.")
parser.add_argument("--CandidateProportion", type=float, default="-1.0", help="Percent for candidate datset.")
parser.add_argument("--SelectorType", type=str, default="-1", help="Query strategy.")
parser.add_argument("--ModelType", type=str, default="-1", help="Predictive model.")
parser.add_argument("--UniqueErrorsInput", type=int, default="-1", help="Unique (1) vs. Duplicate (0) boolean for TreeFarms.")
parser.add_argument("--n_estimators", type=int, default="-1", help="Number of trees for random forest.")
parser.add_argument("--regularization", type=float, default="-1.0", help="Regularization for TreeFarms.")
parser.add_argument("--RashomonThresholdType", type=str, default="-1", help="Adder or multiplier.")
parser.add_argument("--RashomonThreshold", type=float, default="-1.0", help="Rashomon threshold (adder/multiplier) epislon for TreeFarms.")
parser.add_argument("--auto_tune_epsilon", type=ast.literal_eval, default=False, help="True to auto-tune epsilon, False for fixed.")
parser.add_argument("--Type", type=str, default="-1", help="Regression vs. Classification (currently only classification offered).")
parser.add_argument("--DiversityWeight", type=float, default="-1.0", help="Parameter for the weight of diversity in uncertainty metric.")
parser.add_argument("--DensityWeight", type=float, default="-1.0", help="Parameter for the weight of density in uncertainty metric.")
parser.add_argument("--BatchSize", type=int, default="-1", help="Batch size for active learning")
parser.add_argument("--Output", type=str, default="-1", help="Output.")
args = parser.parse_args()

### Parameter Vector ###
SimulationConfigInput = {"DataFileInput": args.Data,
                        "Seed": int(args.Seed),
                        "TestProportion": float(args.TestProportion),
                        "CandidateProportion": float(args.CandidateProportion),
                        "SelectorType": str(args.SelectorType), 
                        "ModelType": str(args.ModelType), 
                        "UniqueErrorsInput": int(args.UniqueErrorsInput),
                        "n_estimators":int(args.n_estimators),
                        "regularization":float(args.regularization),
                        "RashomonThresholdType":args.RashomonThresholdType,
                        "RashomonThreshold":float(args.RashomonThreshold),
                        "Type": args.Type,
                        "DiversityWeight": args.DiversityWeight,
                        "DensityWeight": args.DensityWeight,
                        "BatchSize": args.BatchSize,
                        "auto_tune_epsilon": args.auto_tune_epsilon
                        }

### Run Code ###
SimulationResults = OneIterationFunction(SimulationConfigInput)

### Save Simulation Results ###
with open(os.path.join(SaveDirectory, str(args.Output)), 'wb') as f:
    pickle.dump(SimulationResults, f)