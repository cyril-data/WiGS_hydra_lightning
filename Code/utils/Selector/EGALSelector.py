### Libraries ###
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Exploration Guided Active Learning (EGAL). ###
class EGALSelector:
    """
    Implements Exploration Guided Active Learning (EGAL).
    """

    def __init__(self, gamma=None, **kwargs):
        """
        Args:
            gamma (float): RBF kernel coefficient. If None, defaults to 1/n_features.
        """
        self.gamma = gamma

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Extract Data
        X_train, _ = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")        
        X_train = X_train.values
        X_cand = X_cand.values
        n_features = X_train.shape[1]
        
        gamma = self.gamma if self.gamma is not None else (1.0 / n_features)

        # 2. Calculate Diversity (Dissimilarity to Labeled Set)
        
        if len(X_train) > 0:
            # Distances from Candidates to Training set
            dists_to_train = euclidean_distances(X_cand, X_train)            
            sim_to_train = np.exp(-gamma * (dists_to_train ** 2))            
            max_sim_to_labeled = np.max(sim_to_train, axis=1)            
            diversity_scores = 1.0 - max_sim_to_labeled
        else:
            diversity_scores = np.ones(len(X_cand))

        # 3. Calculate Density (Similarity to Unlabeled Pool)
        dists_pool = euclidean_distances(X_cand, X_cand)
        sim_pool = np.exp(-gamma * (dists_pool ** 2))
        density_scores = (np.sum(sim_pool, axis=1) - 1) / (len(X_cand) - 1)
        
        # Handle edge case where only 1 candidate exists
        if len(X_cand) == 1:
            density_scores = np.array([1.0])

        # 4. Selector
        egal_scores = density_scores * diversity_scores
        best_loc = np.argmax(egal_scores)
        best_index = df_Candidate.iloc[[best_loc]].index[0]

        return {"IndexRecommendation": [float(best_index)]}