# Summary: Computes the Wilcoxon Ranked Signed Test pairwisely for each of the methods in the simulation.
# Input: SimulationErrorResults is a dictionary containing the name of each active learning 
#        method and its values as the error across the active learning process.
# Output: Returns a diagonal matrix whose entries are the p-values from the Wilcoxon Ranked Signed Test 
#         pairwisely comparing each of the methods in the simulation.

### Packages ###
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

### Function ###
def WilcoxonRankSignedTest(SimulationErrorResults, RoundingVal = None):

    ### Set Up ###
    strategies = list(SimulationErrorResults.keys())
    n_strategies = len(strategies)
    PValeMatrix = np.zeros((n_strategies, n_strategies))

    ### Wilcoxon Signed-Rank Test ###
    for i in range(n_strategies):
        for j in range(i):
            stat, pval = wilcoxon(np.mean(SimulationErrorResults[strategies[i]],axis=0), 
                                  np.mean(SimulationErrorResults[strategies[j]],axis=0))
            if RoundingVal == None:
                PValeMatrix[i, j] = pval
            else:
                PValeMatrix[i, j] = np.round(pval,RoundingVal)

    ### Formatting ###
    np.fill_diagonal(PValeMatrix, 1)
    pval_df = pd.DataFrame(PValeMatrix, index=strategies, columns=strategies)
    mask = np.tril(np.ones(pval_df.shape), k=0).astype(bool)
    WRSTResults = pval_df.where(mask, "").astype(str)

    ### Return ###
    return WRSTResults
