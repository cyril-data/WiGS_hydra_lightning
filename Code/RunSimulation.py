### Import Packages ###
import argparse
import math as math
import random as random

### Import functions ###
from utils.Main import *
from utils.Selector import *
from utils.Auxiliary import *
from utils.Prediction import *

### Get Directory ###
cwd = os.getcwd()
SaveDirectory = os.path.join(cwd, "Results")

### Set up argument parser ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, default="-1", help="Job name.")
parser.add_argument("--Data", type=str, default="-1", help="Data type.")
parser.add_argument("--NSim", type=int, default=-1, help="Number of simulations to run.")
parser.add_argument("--ModelType", type=str, default="-1", help="Predictive model.")
parser.add_argument("--TestProportion", type=float, default="-1.0", help="Percent for validation set.")
parser.add_argument("--CandidateProportion", type=float, default="-1.0", help="Percent for candidate datset.")
parser.add_argument("--Output", type=str, default="-1", help="Output.")
args = parser.parse_args()

### Run Code ###
SimulationResults = RunSimulationFunction(DataFileInput = args.Data,
                                          NSim = int(args.NSim),
                                          machine_learning_model = str(args.ModelType),
                                          test_proportion = float(args.TestProportion),
                                          candidate_proportion = float(args.CandidateProportion))

### Save Simulation Results ###
os.makedirs(SaveDirectory, exist_ok=True)
with open(os.path.join(SaveDirectory, str(args.Output)), 'wb') as f:
    pickle.dump(SimulationResults, f)

