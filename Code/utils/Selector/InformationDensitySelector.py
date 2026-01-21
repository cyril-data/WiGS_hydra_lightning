### Libraries ###
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
class InformationDensitySelector:
    """
    Implements Information Density (ID) Selection.
    Score(x) = Uncertainty(x) * Density(x)^beta
    """

    def __init__(self, alpha=0.01, beta=1.0, gamma=0.1, **kwargs):
        """
        Args:
            alpha (float): Ridge regularization strength (matches main model).
            beta (float): Exponent for density importance. Default 1.0.
            gamma (float): Gamma for RBF kernel similarity. Default 0.1.
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Extract Data #
        X_train, _ = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        X_train = X_train.values
        X_cand = X_cand.values
        n_features = X_train.shape[1]
        
        # 2. Uncertainty (Ridge Variance) #
        XTX = np.dot(X_train.T, X_train)
        regularizer = self.alpha * np.eye(n_features)
        try:
            inv_covariance = np.linalg.inv(XTX + regularizer)
        except np.linalg.LinAlgError:
            inv_covariance = np.linalg.pinv(XTX + regularizer)
        intermediate = np.dot(X_cand, inv_covariance)
        variances = np.sum(intermediate * X_cand, axis=1)
        uncertainty_scores = np.sqrt(variances)

        # 3. Density #
        dists = euclidean_distances(X_cand, X_cand)
        similarity_matrix = np.exp(-self.gamma * (dists ** 2))        
        density_scores = np.mean(similarity_matrix, axis=1)

        # 4. Recommendation #
        final_scores = uncertainty_scores * (density_scores ** self.beta)
        best_loc = np.argmax(final_scores)
        best_index = df_Candidate.iloc[[best_loc]].index[0]

        return {"IndexRecommendation": [float(best_index)]}