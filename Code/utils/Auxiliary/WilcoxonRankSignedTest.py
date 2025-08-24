### Packages ###
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import Dict, Optional

### Function ###
def WilcoxonRankSignedTest(SimulationErrorResults: Dict[str, pd.DataFrame],
                           RoundingVal: Optional[int] = None) -> pd.DataFrame:
    """
    Performs a pairwise Wilcoxon signed-rank test on simulation results.

    Args:
        SimulationErrorResults (Dict[str, pd.DataFrame]): A dictionary where each key
            is a strategy name and the value is a pandas DataFrame of its results.
        RoundingVal (int): The number of decimal places to round the p-values to. 
    Returns:
        pd.DataFrame: A formatted DataFrame where the entry at (row i, column j)
        is the p-value from the Wilcoxon test comparing strategy i and strategy j.
    """

    ### Set Up ###
    strategies = list(SimulationErrorResults.keys())
    n_strategies = len(strategies)
    PValeMatrix = np.zeros((n_strategies, n_strategies))

    ### Wilcoxon Signed-Rank Test ###
    for i in range(n_strategies):
        for j in range(i):
            ## Calculate the mean for each simulation ##
            mean_error_i = np.mean(SimulationErrorResults[strategies[i]], axis=0)
            mean_error_j = np.mean(SimulationErrorResults[strategies[j]], axis=0)
            
            ### Perform the Wilcoxon Ranked Signed Test ##
            stat, pval = wilcoxon(mean_error_i, mean_error_j)
            
            if RoundingVal is None:
                PValeMatrix[i, j] = pval
            else:
                PValeMatrix[i, j] = np.round(pval, RoundingVal)

    ### Formatting ###
    np.fill_diagonal(PValeMatrix, 1)
    pval_df = pd.DataFrame(PValeMatrix, index=strategies, columns=strategies)
    mask = np.tril(np.ones(pval_df.shape), k=0).astype(bool)
    WRSTResults = pval_df.where(mask, "").astype(str)

    ### Return ###
    return WRSTResults