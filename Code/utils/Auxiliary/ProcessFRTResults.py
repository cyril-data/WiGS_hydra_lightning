### Import libraries ###
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

### Process Files in Batches Function ###
def ProcessBatch(files, batch_size=5):
    ### Initialize Storage ###
    ThresholdValuesStorage = []
    Epsilon_F1ScoreStorage = []
    Epsilon_ClassAccuracyStorage = []
    
    ### Process in Batches ###
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        ### Process Current Batch ###
        for file in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}"):
            try:
                ### Load and Immediately Process File ###
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    
                    # Store data while clearing memory
                    ThresholdValuesStorage.append(data["ThresholdValues"])
                    Epsilon_F1ScoreStorage.append(data["Epsilon_F1Score"])
                    Epsilon_ClassAccuracyStorage.append(data["Epsilon_ClassAccuracy"])
                    
                    # Clear data from memory
                    del data
                    
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
    
    ### Convert to DataFrames ###
    ThresholdValuesDF = pd.DataFrame(ThresholdValuesStorage)
    Epsilon_F1ScoreDF = pd.DataFrame(Epsilon_F1ScoreStorage)
    Epsilon_ClassAccuracyDF = pd.DataFrame(Epsilon_ClassAccuracyStorage)
    
    return ThresholdValuesDF, Epsilon_F1ScoreDF, Epsilon_ClassAccuracyDF

### Calculate Mean Values Function ###
def CalculateMeanValues(ProcessedDirectory):
    ### Load CSV Files ###
    ThresholdValuesALL = pd.read_csv(os.path.join(ProcessedDirectory, "ThresholdValuesStorage.csv"))
    Epsilon_F1ScoreALL = pd.read_csv(os.path.join(ProcessedDirectory, "Epsilon_F1ScoreStorage.csv"))
    Epsilon_ClassAccuracyALL = pd.read_csv(os.path.join(ProcessedDirectory, "Epsilon_ClassAccuracyStorage.csv"))
    
    ### Shape ###
    print("ThresholdValuesALL: " + str(ThresholdValuesALL.shape[0]))
    print("Epsilon_F1ScoreALL: " + str(Epsilon_F1ScoreALL.shape[0]))
    print("Epsilon_ClassAccuracyALL: " + str(Epsilon_ClassAccuracyALL.shape[0]))

    ### Calculate Mean Values ###
    ThresholdValues_Mean = np.mean(ThresholdValuesALL, axis=0)
    Epsilon_F1Score_Mean = np.mean(Epsilon_F1ScoreALL, axis=0)
    Epsilon_ClassAccuracy_Mean = np.mean(Epsilon_ClassAccuracyALL, axis=0)
    
    ### Create Dictionary ###
    MeanValues = {
        "ThresholdValues_Mean": ThresholdValues_Mean,
        "Epsilon_F1Score_Mean": Epsilon_F1Score_Mean,
        "Epsilon_ClassAccuracy_Mean": Epsilon_ClassAccuracy_Mean
    }
    
    return MeanValues

### Main Function ###
def main():
    ### Parser ###
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument("--DataType", type=str, required=True, help="Type of data.")
    args = parser.parse_args()

    ### Set Up Directories ###
    cwd = os.getcwd()
    ResultsDirectory = os.path.join(cwd, "Results")
    OutputDirectory = os.path.join(ResultsDirectory, "OptimalThreshold")
    RawDirectory = os.path.join(ResultsDirectory, "OptimalThreshold", args.DataType, "Raw")
    ProcessedDirectory = os.path.join(OutputDirectory, args.DataType, "Processed")
    
    ### Create Processed Directory if it Doesn't Exist ###
    os.makedirs(ProcessedDirectory, exist_ok=True)

    ### Extract File Names ###
    CategoryFileNames = [os.path.join(RawDirectory, filename) 
                        for filename in os.listdir(RawDirectory) 
                        if filename.endswith('.pkl')]

    ### Process Files in Batches ###
    print(f"Processing {len(CategoryFileNames)} files for {args.DataType}...")
    ThresholdValuesStorage, Epsilon_F1ScoreStorage, Epsilon_ClassAccuracyStorage = ProcessBatch(
        CategoryFileNames, 
        batch_size=5
    )
    
    ### Save Individual Results ###
    print(f"Saving individual results for {args.DataType}...")
    ThresholdValuesStorage.to_csv(os.path.join(ProcessedDirectory, "ThresholdValuesStorage.csv"), index=False)
    Epsilon_F1ScoreStorage.to_csv(os.path.join(ProcessedDirectory, "Epsilon_F1ScoreStorage.csv"), index=False)
    Epsilon_ClassAccuracyStorage.to_csv(os.path.join(ProcessedDirectory, "Epsilon_ClassAccuracyStorage.csv"), index=False)

    ### Calculate and Save Mean Values ###
    print(f"Calculating and saving mean values for {args.DataType}...")
    MeanValues = CalculateMeanValues(ProcessedDirectory)
    
    ### Save Mean Values ###
    with open(os.path.join(ProcessedDirectory, "FRTMeanValues.pkl"), 'wb') as file:
        pickle.dump(MeanValues, file)

    print(f"Successfully saved all {args.DataType} files, including mean values!")

### Run Main ###
if __name__ == "__main__":
    main()