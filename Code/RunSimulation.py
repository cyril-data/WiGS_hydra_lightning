### Import Packages ###
import argparse
import os
import pickle

### Import functions ###
from utils.Main.RunSimulationFunction import RunSimulationFunction

### Get Directory ###
CWD = os.getcwd() 
PROJECT_ROOT = os.path.dirname(CWD) 
SAVE_DIRECTORY = os.path.join(PROJECT_ROOT, "Results", "simulation_results")

### Set up argument parser ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--JobName", type=str, help="Job name.")
parser.add_argument("--Data", type=str, help="Data type.")
parser.add_argument("--NSim", type=int, help="Number of simulations to run.")
parser.add_argument("--ModelType", type=str, help="Predictive model.")
parser.add_argument("--TestProportion", type=float, help="Percent for validation set.")
parser.add_argument("--CandidateProportion", type=float, help="Percent for candidate dataset.")
parser.add_argument("--Output", type=str, help="Output filename.")
args = parser.parse_args()

### Run Code ###
SimulationResults = RunSimulationFunction(
    DataFileInput=args.Data,
    NSim=int(args.NSim),
    machine_learning_model=str(args.ModelType),
    test_proportion=float(args.TestProportion),
    candidate_proportion=float(args.CandidateProportion)
)

### Save Simulation Results ###
os.makedirs(SAVE_DIRECTORY, exist_ok=True) 
output_path = os.path.join(SAVE_DIRECTORY, str(args.Output))
with open(output_path, 'wb') as f:
    pickle.dump(SimulationResults, f)