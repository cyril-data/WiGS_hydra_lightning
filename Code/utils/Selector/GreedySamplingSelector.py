
### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target 

class GreedySamplingSelector:
    """
    Implements the greedy sampling methods from Wu, Lin, and Huang (2018).

    Attributes:
        strategy (str): The active strategy to be used ('GSx', 'GSy', or 'iGS').
        distance (str): The distance metric used for calculations (e.g., 'euclidean').
        Seed (int): The random seed for reproducibility (Note: not used in current
            implementation but retained for API consistency).
    """

    ### Initialize ###
    def __init__(self, strategy: str, distance: str = "euclidean", Seed: int = None, **kwargs):
        """
        Initializes the GreedySamplingSelector.

        Args:
            strategy (str): The greedy sampling strategy. Must be one of 'GSx', 'GSy', or 'iGS'.
            distance (str, optional): The distance metric to use, compatible with `scipy.spatial.distance.cdist`. 
            Seed (int, optional): A random seed for reproducibility.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """
        if strategy not in ['GSx', 'GSy', 'iGS']:
            raise ValueError(f"Invalid greedy sampling strategy: {strategy}. Must be 'GSx', 'GSy', or 'iGS'.")
        self.strategy = strategy
        self.distance = distance
        self.Seed = Seed 

    ### Select Observation ###
    def select(self, 
               df_Candidate: pd.DataFrame, 
               Model=None, 
               df_Train: pd.DataFrame = None, 
               **kwargs) -> dict:
        
        """
        Selects the single most informative observation from the candidate set.

        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points from which to select.
            Model (object, optional): A trained model.
            df_Train (pd.DataFrame, optional): The current set of labeled training data.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.

        Returns:
            dict: A dictionary containing the recommended point's index, in the format `{'IndexRecommendation': [index]}`.
        """
        
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Set up candidate features ##
        X_Candidate, _ = get_features_and_target(
            df=df_Candidate, target_column_name="Y")
        X_Candidate_np = X_Candidate.values

        ## Set up training features ##
        X_Train, y_Train = get_features_and_target(
            df=df_Train, target_column_name="Y")
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
            if d_nmX is None or d_nmY is None: 
                raise RuntimeError("iGS strategy requires both GSx and GSy components.")
            d_nXY_matrix = d_nmX * d_nmY
            d_nXY = d_nXY_matrix.min(axis=1) 
            MaxRowNumber = np.argmax(d_nXY)

        ## Output ##
        IndexRecommendation_int = int(df_Candidate.iloc[[MaxRowNumber]].index[0])
        return {"IndexRecommendation": IndexRecommendation_int}