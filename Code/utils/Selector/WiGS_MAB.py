### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class WiGS_MAB_Selector:

    ### Initialize ###
    def __init__(self,
                 mab_arms: list = None,
                 mab_c: float = 2.0,
                 **kwargs):
        
        ## Set up MAB ##
        self.arms = mab_arms if mab_arms is not None else [0.25, 0.5, 0.75]
        self.mab_c = mab_c 
        
        # MAB state variables
        self.arm_counts = np.zeros(len(self.arms))
        self.arm_values = np.zeros(len(self.arms))
        self.iteration = 0
        self.last_arm_pulled = None
        self.last_rmse = None
        
    ### Select Observation ###
    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, 
               auxiliary_columns: list = None, current_rmse: float = None) -> dict:
        
        # --- MAB Part 1: Update values based on the reward from the LAST step ---
        if self.last_arm_pulled is not None and self.last_rmse is not None:
            if current_rmse is None:
                raise ValueError("MAB strategy requires 'current_rmse' to be passed.")
            reward = self.last_rmse - current_rmse
            
            last_arm_idx = self.last_arm_pulled
            old_value = self.arm_values[last_arm_idx]
            new_count = self.arm_counts[last_arm_idx]
            self.arm_values[last_arm_idx] = old_value + (reward - old_value) / new_count

        self.last_rmse = current_rmse

        # --- Shared Logic: Calculate distances and normalize ---
        if df_Candidate.empty:
            return {"IndexRecommendation": []}
        
        X_Candidate, _ = get_features_and_target(df=df_Candidate, target_column_name="Y", auxiliary_columns=auxiliary_columns)
        X_Train, y_Train = get_features_and_target(df=df_Train, target_column_name="Y", auxiliary_columns=auxiliary_columns)
        d_nmX = cdist(X_Candidate.values, X_Train.values, metric='euclidean')
        Predictions = Model.predict(X_Candidate)
        d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric='euclidean')
        
        epsilon = 1e-8
        d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        # --- MAB Part 2: Select an arm for the CURRENT step using UCB1 ---
        arm_to_pull = -1
        if self.iteration < len(self.arms):
            arm_to_pull = self.iteration
        else:
            # Add a small epsilon to arm_counts to avoid division by zero
            safe_arm_counts = self.arm_counts + epsilon
            exploration_bonus = self.mab_c * np.sqrt(np.log(self.iteration) / safe_arm_counts)
            ucb_scores = self.arm_values + exploration_bonus
            arm_to_pull = np.argmax(ucb_scores)
        
        w_x = self.arms[arm_to_pull]
        w_y = 1.0 - w_x
        
        self.last_arm_pulled = arm_to_pull
        self.arm_counts[arm_to_pull] += 1

        # --- Final Score and Selection ---
        score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        final_scores = score_matrix.min(axis=1)
        best_candidate_iloc = np.argmax(final_scores)
        self.iteration += 1

        IndexRecommendation = df_Candidate.iloc[[best_candidate_iloc]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)]}