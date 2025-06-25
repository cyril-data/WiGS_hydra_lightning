# Summary: A python script to extract the error and time for the active learning simulation. It accesses each of the .pkl result files
#          from the simulations, and places each error (time) into a row in the ErrorMatrix.csv (TimeMatrix.csv) file.
# Input: 
#   DataType: A string that indicates either "Simulate" for the simulation or the name of the DataFrame in the Data folder.
#   ModelType: Predictive model. Examples can be LinearRegression or RandomForestRegresso.
#   Categories: The last identifying portion of the results file.
#               For the DUREAL, UNREAL, and RandomForests methods, the respective inputs are
#               {"MTTreeFarms_UEI0_NE100_Reg0.01_RBA0.01.pkl", 
#                "MTTreeFarms_UEI1_NE100_Reg0.01_RBA0.01.pkl",
#                "MTRandomForestClassification_UEI0_NE100_Reg0.01_RBA0.01.pkl"}
# Output: Outputs the matrices ErrorMatrix and TimeMatrix into the ProcessedResults folder.

### Import libraries ###
import os
import pickle
import argparse
import numpy as np
import pandas as pd

### Extract Error and Time Function ###
def ExtractInformation(files):

    ### Set Up ###
    TimeVec = []
    ErrorVec = []
    EpsilonVec = []
    AllTreeCountVec = []
    RefitDecisionVec = [] 
    UniqueTreeCountVec = []
    SelectionHistoryVec = []
    for file in files:
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                ErrorVec.append(data["ErrorVec"])
                TimeVec.append(data["ElapsedTime"])
                SelectionHistoryVec.append(data["SelectionHistory"])
                AllTreeCountVec.append(data["TreeCount"]["AllTreeCount"])
                UniqueTreeCountVec.append(data["TreeCount"]["UniqueTreeCount"])

                if "EpsilonVec" in data and data["EpsilonVec"] is not None:
                    EpsilonVec.append(data["EpsilonVec"])
                else:
                    if ErrorVec:
                        num_iterations = len(ErrorVec[-1])
                        EpsilonVec.append(np.full(num_iterations, np.nan))
                    else:
                        EpsilonVec.append(np.array([]))
                
                # <--- NEW: Safely extract RefitDecisionVec if it exists in the data
                if "RefitDecisionVec" in data and data["RefitDecisionVec"] is not None:
                    RefitDecisionVec.append(data["RefitDecisionVec"])
                else:
                    if ErrorVec: # Use ErrorVec length as a fallback for iterations
                        num_iterations = len(ErrorVec[-1])
                        RefitDecisionVec.append(np.full(num_iterations, np.nan))
                    else:
                        RefitDecisionVec.append(np.array([]))

        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    return np.array(ErrorVec), np.array(TimeVec), list(SelectionHistoryVec)[0], np.array(AllTreeCountVec), np.array(UniqueTreeCountVec), np.array(EpsilonVec), np.array(RefitDecisionVec)

### Parser ###
parser = argparse.ArgumentParser(description="Aggregate simulation results.")
parser.add_argument("--DataType", type=str, required=True, help="Type of data.")
parser.add_argument("--ModelType", type=str, required=True, help="Prediction model type.")
parser.add_argument("--Categories", type=str, required=True, help="Single category string.")
args = parser.parse_args()

### Set Up ###
cwd = os.getcwd()
ResultsDirectory = os.path.join(cwd, "Results", args.DataType, args.ModelType)
OutputDirectory = os.path.join(ResultsDirectory, "ProcessedResults")
RawDirectory = os.path.join(ResultsDirectory, "Raw")
Category = args.Categories

### Extract File Names ###
CategoryFileNames = []
for filename in os.listdir(RawDirectory):
    if filename.endswith(".pkl") and Category in filename:
        CategoryFileNames.append(os.path.join(RawDirectory, filename))

### Extract Data ###
if not CategoryFileNames:
    print(f"Warning: No files found for category {Category}. Exiting.")
    exit(1)
print(f"Processing category: {Category} with {len(CategoryFileNames)} files")
ErrorVec, TimeVec, SelectionHistoryVec, AllTreeCountVec, UniqueTreeCountVec, EpsilonVec, RefitDecisionVec = ExtractInformation(CategoryFileNames)
ErrorMatrix = pd.DataFrame(ErrorVec.squeeze())
TimeMatrix = pd.DataFrame(TimeVec.squeeze())
AllTreeCountVec = pd.DataFrame(AllTreeCountVec.squeeze())
UniqueTreeCountVec = pd.DataFrame(UniqueTreeCountVec.squeeze())
EpsilonMatrix = pd.DataFrame(EpsilonVec.squeeze())
RefitDecisionMatrix = pd.DataFrame(RefitDecisionVec.squeeze())


### Save ###
os.makedirs(os.path.join(OutputDirectory, "ErrorVec"), exist_ok=True)
os.makedirs(os.path.join(OutputDirectory, "ElapsedTime"), exist_ok=True)
os.makedirs(os.path.join(OutputDirectory, "TreeCount"), exist_ok=True)
os.makedirs(os.path.join(OutputDirectory, "SelectionHistory"), exist_ok=True)
os.makedirs(os.path.join(OutputDirectory, "EpsilonVec"), exist_ok=True)
os.makedirs(os.path.join(OutputDirectory, "RefitDecisionVec"), exist_ok=True)

ErrorMatrix.to_csv(os.path.join(OutputDirectory, "ErrorVec", f"{Category}_ErrorMatrix.csv"), index=False)
TimeMatrix.to_csv(os.path.join(OutputDirectory, "ElapsedTime", f"{Category}_TimeMatrix.csv"), index=False)
AllTreeCountVec.to_csv(os.path.join(OutputDirectory, "TreeCount", f"{Category}_AllTreeCount.csv"), index=False)
UniqueTreeCountVec.to_csv(os.path.join(OutputDirectory, "TreeCount", f"{Category}_UniqueTreeCount.csv"), index=False)
EpsilonMatrix.to_csv(os.path.join(OutputDirectory, "EpsilonVec", f"{Category}_EpsilonMatrix.csv"), index=False)
RefitDecisionMatrix.to_csv(os.path.join(OutputDirectory, "RefitDecisionVec", f"{Category}_RefitDecisionMatrix.csv"), index=False) # <--- NEW: Save RefitDecisionMatrix


with open(os.path.join(OutputDirectory, "SelectionHistory", f"{Category}_SelectionHistory.pkl"), 'wb') as file:
    pickle.dump(SelectionHistoryVec, file)
print(f"Saved {Category} files!")
