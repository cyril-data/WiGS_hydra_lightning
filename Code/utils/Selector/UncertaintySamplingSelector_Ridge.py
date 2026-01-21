### Library ###
import numpy as np
import pandas as pd
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
class UncertaintySamplingSelector_Ridge:
    """
    Selects the candidate point x that maximizes the model's predictive variance for Ridge Regression. 
    """

    def __init__(self, alpha=0.01, **kwargs):
        """
        Args:
            alpha (float): Regularization strength. MUST match the main model's alpha.
                           Default is 0.01 to match your RidgeRegressionPredictor.
            **kwargs: Ignored arguments.
        """
        self.alpha = float(alpha)

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        """
        Selects the candidate with the highest analytical predictive variance.
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ### Matrix algebra ###
        X_train, _ = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")        
        X_train = X_train.values
        X_cand = X_cand.values
        n_features = X_train.shape[1]
        XTX = np.dot(X_train.T, X_train)
        regularizer = self.alpha * np.eye(n_features)
        try:
            inv_covariance = np.linalg.inv(XTX + regularizer)
        except np.linalg.LinAlgError:
            inv_covariance = np.linalg.pinv(XTX + regularizer)
        intermediate = np.dot(X_cand, inv_covariance)

        ### Variance ###
        variances = np.sum(intermediate * X_cand, axis=1)  

        ### Selection ####      
        best_loc = np.argmax(variances)
        best_index = df_Candidate.iloc[[best_loc]].index[0]
        return {"IndexRecommendation": [float(best_index)]}