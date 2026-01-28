### Libraries ###
import numpy as np
import pandas as pd
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### EMCM ###
class EMCMSelector:
    """
    Implements Expected Model Change Maximization (EMCM) for Ridge Regression.
    """

    def __init__(self, alpha=0.01, **kwargs):
        """
        Args:
            alpha (float): Regularization strength. Matches RidgeRegressionPredictor.
        """
        self.alpha = float(alpha)

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Extract Data
        X_train, _ = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        X_train = X_train.values
        X_cand = X_cand.values
        n_features = X_train.shape[1]
        
        # 2. Compute Inverse Covariance (Precision Matrix)
        XTX = np.dot(X_train.T, X_train)
        regularizer = self.alpha * np.eye(n_features)
        
        try:
            inv_covariance = np.linalg.inv(XTX + regularizer)
        except np.linalg.LinAlgError:
            inv_covariance = np.linalg.pinv(XTX + regularizer)
            
        # 3. Calculate Components efficiently
        update_directions = np.dot(X_cand, inv_covariance)
        update_norms = np.linalg.norm(update_directions, axis=1)
        variances = np.sum(update_directions * X_cand, axis=1)
        sigmas = np.sqrt(variances)
        
        # 4. Calculate EMCM Score
        emcm_scores = update_norms * sigmas
        
        # 5. Select Max
        best_loc = np.argmax(emcm_scores)
        best_index = df_Candidate.iloc[[best_loc]].index[0]

        return {"IndexRecommendation": [float(best_index)]}