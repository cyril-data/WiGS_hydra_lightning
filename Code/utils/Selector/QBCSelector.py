### Libraries ###
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import resample
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Query-By-Bagging (QBB) / Query-By-Committee (QBC) ###
class QBCSelector:
    """
    Implements Query-By-Bagging (QBB) / Query-By-Committee (QBC).
    
    The 'committee' consists of multiple models trained on bootstrap samples
    of the labeled data. The committee members are cloned from the main
    predictor model, ensuring consistency with whatever model is used in
    the active learning loop (e.g., Ridge, Gaussian Process, etc.).
    
    The acquisition function selects the candidate point with the highest
    variance in predictions across the committee members.
    """

    def __init__(self, n_committee=5, seed=None, **kwargs):
        """
        Args:
            n_committee (int): Number of models in the committee.
            seed (int): Random seed for reproducibility.
            **kwargs: Ignored arguments.
        """
        self.n_committee = int(n_committee)
        self.seed = seed

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, Model=None, **kwargs) -> dict:
        """
        Selects the candidate with the highest prediction variance.

        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points.
            df_Train (pd.DataFrame): The current set of labeled training data.
            Model (object): The main predictor model. Its underlying sklearn
                estimator (.model) is cloned to build each committee member.
            **kwargs: Ignored arguments.

        Returns:
            dict: A dictionary containing the recommended point's index.
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Prepare Data
        X_train, y_train = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        # 2. Train Committee by cloning the main model's sklearn estimator
        base_estimator = Model.model
        predictions = []

        for i in range(self.n_committee):
            member_seed = self.seed + i if self.seed is not None else None
            X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=member_seed)
            
            member = clone(base_estimator)
            member.fit(X_boot, y_boot)
            preds = member.predict(X_cand)
            predictions.append(preds)

        # 3. Calculate Variance across Committee
        committee_preds = np.vstack(predictions)        
        prediction_variance = np.var(committee_preds, axis=0)
        
        # 4. Select Max Variance
        best_idx_loc = np.argmax(prediction_variance)
        IndexRecommendation = df_Candidate.iloc[[best_idx_loc]].index[0]

        return {"IndexRecommendation": [float(IndexRecommendation)]}