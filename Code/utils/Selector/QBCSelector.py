import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class QBCSelector:
    """
    Implements Heterogeneous Query-By-Committee (QBC).
    
    The 'committee' consists of distinct model architectures representing
    different inductive biases. All members are trained on the FULL current
    dataset (no bootstrapping).
    """

    def __init__(self, seed=None, **kwargs):
        """
        Args:
            seed (int): Random seed for reproducibility.
            **kwargs: Ignored arguments.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Define Committee #
        self.committee_members = [
            LinearRegression(),                                                    # 1. Linear Baseline
            DecisionTreeRegressor(max_depth=5, random_state=seed),                 # 2. Single Tree (Step functions)
            KNeighborsRegressor(n_neighbors=5),                                    # 3. Instance-based (Local averaging)
            SVR(kernel='rbf', C=1.0),                                              # 4. SVR (Smooth curves, Kernel)
            GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=seed) # 5. Boosting (Ensemble)
        ]

    def select(self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, **kwargs) -> dict:
        """
        Selects the candidate with the highest prediction variance across 
        the heterogeneous committee.
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Prepare Data
        X_train, y_train = get_features_and_target(df_Train, "Y")
        X_cand, _ = get_features_and_target(df_Candidate, "Y")
        
        # 2. Train Committee
        predictions = [] # Shape: (n_members, n_candidates)
        
        for model in self.committee_members:
            # Train on FULL current data
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_cand)
            predictions.append(preds)

        # 3. Calculate Variance across Committee
        committee_preds = np.vstack(predictions)
        
        # Variance across the different algorithms
        prediction_variance = np.var(committee_preds, axis=0)
        
        # 4. Select Max Variance
        best_idx_loc = np.argmax(prediction_variance)
        IndexRecommendation = df_Candidate.iloc[[best_idx_loc]].index[0]

        return {"IndexRecommendation": [float(IndexRecommendation)]}