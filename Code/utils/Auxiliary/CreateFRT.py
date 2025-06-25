### Import packages ###
import os
import argparse
import itertools
import numpy as np
import pandas as pd

### Set up argument parser ###
parser = argparse.ArgumentParser(description="Parse command line arguments for job parameters")
parser.add_argument("--DataType", type=str, default="-1", help="Simulation case number.")
parser.add_argument("--RashomonThreshold", type=str, default="-1", help="Simulation case number.")
args = parser.parse_args()

### Directory ###
cwd = os.getcwd()
# ParentDirectory = os.path.abspath(os.path.join(cwd, "../.."))

# Input Parameters #
ParameterDictionary = {"Data":[args.DataType],
                       "Seed":list(range(0,100)),
                       "TestProportion":[0.25],
                    #    "CandidateProportion":[0.8],
                       "regularization": [0.01],
                       "RashomonThresholdType": ["Adder"],                                         # ["Adder", "Multiplier"]
                       "RashomonThreshold": [args.RashomonThreshold],
                       "Partition": ["short"],                                                        # [short, medium, long, largemem, compute, cpu-g2-mem2x]
                       "Time": ["11:59:00"],                                                            # [00:59:00, 11:59:00, 6-23:59:00]
                       "Memory": ["30000M"]}                                                                # [100M, 30000M, 100000M]

# Create Parameter Vector #
ParameterVector = pd.DataFrame.from_records(itertools.product(*ParameterDictionary.values()), columns=ParameterDictionary.keys())


# Generate JobName #
ParameterVector["JobName"] = (
    ParameterVector["Seed"].astype(str) +
    ParameterDictionary["Data"] + "_FRT")

# Generate OutputName #
ParameterVector["Output"] = ParameterVector["Seed"].astype(str) + ParameterVector["Data"].astype(str) +"_FRT" + ".pkl"

### Loop through each row in the DataFrame ###
for i, row in ParameterVector.iterrows():
    
    ## Extract parameters ###
    JobName = row['JobName']
    Data = row['Data']
    Seed = row['Seed']
    TestProportion = row['TestProportion']
    # CandidateProportion = row['CandidateProportion']
    regularization = row['regularization']
    RashomonThresholdType = row['RashomonThresholdType']
    RashomonThreshold = row['RashomonThreshold']
    Output = row['Output']
    Partition = row["Partition"]
    Time = row["Time"]
    Memory = row["Memory"]
    
    # Path for .sbatch files ###
    TargetDirectory = os.path.join(cwd,"Code", "Cluster", "OptimalValues", Data, "RunFRT")
    sbatch_file_path = os.path.join(TargetDirectory, f"{JobName}.sbatch")
    
    # Create the .sbatch file content
    sbatch_content = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JobName}",
        f"#SBATCH --partition={Partition}",                                             # [short, medium, long, largemem, or compute]
        "#SBATCH --ntasks=1",
        f"#SBATCH --time={Time}",                                                # [11:59:00, 6-23:59:00]
        f"#SBATCH --mem-per-cpu={Memory}",                                             # [30000, 100000]
        f"#SBATCH -o ClusterMessages/out/myscript_{JobName}_%j.out",
        f"#SBATCH -e ClusterMessages/error/myscript_{JobName}_%j.err",
        "#SBATCH --mail-type=FAIL",                                             # FAIL ALL
        "#SBATCH --mail-user=simondn@uw.edu",
        "",
        "cd ~/RashomonActiveLearning",
        "module load Python",
        "python Code/OptimalThresholdSimulation.py \\",
        f"    --JobName " + JobName +" \\",
        f"    --Data {Data} \\",
        f"    --Seed {Seed} \\",
        f"    --TestProportion {TestProportion} \\",
        # f"    --CandidateProportion {CandidateProportion} \\",
        f"    --regularization {regularization} \\",
        f"    --RashomonThresholdType {RashomonThresholdType} \\",
        f"    --RashomonThreshold {RashomonThreshold} \\",
        f"    --Output {Output}"
    ]

    # Write content to .sbatch file
    os.makedirs(os.path.dirname(sbatch_file_path), exist_ok=True)  # Ensure directory exists
    with open(sbatch_file_path, "w") as sbatch_file:
        sbatch_file.write("\n".join(sbatch_content))

print("Creation Sbatch files generated successfully.")
