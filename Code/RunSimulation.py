### Import packages ###
import argparse
import os
import pickle
from utils.Main.RunSimulationFunction import RunSimulationFunction

### Models ###
MODEL_LIST = [
    'RidgeRegressionPredictor'
]

### Get Directory Paths ###
CWD = os.getcwd() 
PROJECT_ROOT = os.path.dirname(CWD) 
BASE_SAVE_DIRECTORY = os.path.join(PROJECT_ROOT, "Results", "simulation_results", "raw")

### Set up argument parser ###
parser = argparse.ArgumentParser(description="Parse command line arguments for a single job task.")
parser.add_argument("--Data", type=str, required=True, help="Data type for this job array.")
parser.add_argument("--TaskID", type=int, required=True, help="SLURM Array Task ID (1-based).")
parser.add_argument("--NReplications", type=int, required=True, help="Total number of simulations per model.")
parser.add_argument("--CandidateProportion", type=float, required=True, help="Percent for candidate dataset.")
args = parser.parse_args()

### Map Task ID to Model and Seed ###
task_id_zero_based = args.TaskID - 1
model_index = task_id_zero_based // args.NReplications
replication_seed = task_id_zero_based % args.NReplications

try:
    model_type = MODEL_LIST[model_index]
except IndexError:
    print(f"Error: TaskID {args.TaskID} resulted in an invalid model index.")
    exit(1)

### Create new nested save directory and unique filename ###
data_save_dir = os.path.join(BASE_SAVE_DIRECTORY, args.Data)
os.makedirs(data_save_dir, exist_ok=True) 
output_filename = f"{args.Data}_{model_type}_seed_{replication_seed}.pkl"
output_path = os.path.join(data_save_dir, output_filename)
print(f"--- Starting Task {args.TaskID}: Dataset={args.Data}, Model={model_type}, Seed={replication_seed} ---")

### Run the Simulation for a Single Seed and Model ###
SimulationResults = RunSimulationFunction(
    DataFileInput=args.Data,
    Seed=replication_seed,
    machine_learning_model=model_type,
    candidate_proportion=float(args.CandidateProportion)
)

### Save Simulation Results to the new nested directory ###
with open(output_path, 'wb') as f:
    pickle.dump(SimulationResults, f)

print(f"--- Task {args.TaskID} Finished: Saved results to {output_path} ---")