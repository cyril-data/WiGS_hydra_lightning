# Summary: 
# Input: 
# Output

### Libraries ###
import pandas as pd
import numpy as np
from utils.Auxiliary.LoadDataSet import LoadData

def SelectionHistoryRankFunction(SelectionHistory, DataType):

   ### Set Up ###
    ObservedPositions = {}

    ### Iterate over simulation and rows ###
    for sim_idx, row in SelectionHistory.iterrows():
        for iteration_idx, observation in enumerate(row):
            if observation not in ObservedPositions:
                ObservedPositions[observation] = []
            ObservedPositions[observation].append(iteration_idx + 1)

    ### Compute Average Rank ###
    AverageRank = {}
    for observation, ranks in ObservedPositions.items():
        AverageRank[observation] = np.mean(ranks)

    ### Reformat ###
    AverageRankOutput = pd.DataFrame(list(AverageRank.items()), columns=['Observation', 'AverageRank'])
    AverageRankOutput.index = AverageRankOutput["Observation"].values
    AverageRankOutput = AverageRankOutput.drop(AverageRankOutput.columns[0], axis=1)
    AverageRankOutput = AverageRankOutput.join(LoadData(DataType), how='left')

    ### Return ###
    return(AverageRankOutput)
