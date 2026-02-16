### Libraries ###
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.utils import resample
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Query-By-Bagging (QBB) / Query-By-Committee (QBC) ###
class QBCSelector:
    """
    Implements Query-By-Bagging (QBB) / Query-By-Committee (QBC) with Ridge Regression.
    
    The 'committee' consists of multiple Ridge Regression models trained on 
    bootstrap samples of the labeled data. The acquisition function selects 
    the candidate point with the highest variance in predictions across the 
    committee members.
    """

    def __init__(self, n_committee=5, alpha=0.01, seed=None, **kwargs):
        """
        Args:
            n_committee (int): Number of models in the committee.
            alpha (float): Regularization strength for the Ridge members.
            seed (int): Random seed for reproducibility.
            **kwargs: Ignored arguments.
        """
        self.n_committee = int(n_committee)
        self.alpha = float(alpha)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        """
        Selects the candidate with the highest prediction variance.
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Prepare Data
        X_train, y_train = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        # 2. Train Committee
        predictions = [] # Shape: (n_committee, n_candidates)
        
        for i in range(self.n_committee):
            member_seed = self.seed + i if self.seed is not None else None            
            X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=member_seed)            
            model = Ridge(alpha=self.alpha)
            model.fit(X_boot, y_boot)
            preds = model.predict(X_cand)
            predictions.append(preds)

        # 3. Calculate Variance across Committee
        committee_preds = np.vstack(predictions)        
        prediction_variance = np.var(committee_preds, axis=0)
        
        # 4. Select Max Variance
        best_idx_loc = np.argmax(prediction_variance)
        IndexRecommendation = df_Candidate.iloc[[best_idx_loc]].index[0]

        return {"IndexRecommendation": [float(IndexRecommendation)]}