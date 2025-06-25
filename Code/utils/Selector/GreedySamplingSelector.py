# Summary: Implements the greedy sampling methods from Wu, Lin, and Huang (2018).
#          GSx samples based on the covariate space, GSy based on the output space, and iGS on both.

### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target # Ensure this is imported

class GreedySamplingSelector:

    ### Initialize ###
    def __init__(self, strategy: str, distance: str = "euclidean", BatchSize: int = 1, Seed: int = None, **kwargs):
        if strategy not in ['GSx', 'GSy', 'iGS']:
            raise ValueError(f"Invalid greedy sampling strategy: {strategy}. Must be 'GSx', 'GSy', or 'iGS'.")
        self.strategy = strategy
        self.distance = distance
        self.BatchSize = BatchSize
        self.Seed = Seed 

    ### Select Observation ###
    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, auxiliary_columns: list = None) -> dict:
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Set up candidate features ##
        X_Candidate, _ = get_features_and_target(
            df=df_Candidate, target_column_name="Y", auxiliary_columns=auxiliary_columns)
        X_Candidate_np = X_Candidate.values

        ## Set up training features ##
        X_Train, y_Train = get_features_and_target(
            df=df_Train, target_column_name="Y", auxiliary_columns=auxiliary_columns)
        X_Train_np = X_Train.values

        ## GSx Logic ##
        d_nX = None
        if self.strategy in ['GSx', 'iGS']:
            d_nmX = cdist(X_Candidate_np, X_Train_np, metric=self.distance)
            d_nX = d_nmX.min(axis=1) 

        ## GSy Logic ##
        d_nY = None
        if self.strategy in ['GSy', 'iGS']:
            Predictions = Model.predict(X_Candidate) 
            d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric=self.distance)
            d_nY = d_nmY.min(axis=1) 

        ## Selection ##
        MaxRowNumber = -1
        if self.strategy == 'GSx':
            MaxRowNumber = np.argmax(d_nX)
        elif self.strategy == 'GSy':
            MaxRowNumber = np.argmax(d_nY)
        elif self.strategy == 'iGS':
            if d_nX is None or d_nY is None:
                raise RuntimeError("iGS strategy requires both GSx and GSy components, but one was not computed.")
            d_nXY = d_nX * d_nY
            MaxRowNumber = np.argmax(d_nXY)

        ## Output ##
        IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)]}