### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class WeightedGreedySamplingSelector:

    ### Initialize ###
    def __init__(self,
                 weight_strategy: str,
                 initial_candidate_size: int = None,
                 w_x: float = 0.5,
                 decay_type: str = 'linear',
                 decay_constant: float = 5.0,
                 **kwargs):
        
        ## Set up ##
        self.weight_strategy = weight_strategy
        self.iteration = 0 

        ## Static input weights  ##
        if self.weight_strategy == 'static':
            if not (0 <= w_x <= 1):
                raise ValueError("For 'static' strategy, w_x must be between 0 and 1.")
            self.w_x = w_x
            self.w_y = 1 - w_x
        
        ## Time decayed weights ##
        elif self.weight_strategy == 'time_decay':
            if initial_candidate_size is None or initial_candidate_size <= 0:
                raise ValueError("For 'time_decay' strategy, initial_candidate_size must be a positive integer.")
            self.initial_candidate_size = initial_candidate_size
            self.decay_type = decay_type
            self.decay_constant = decay_constant
        
        else:
            raise ValueError(f"Invalid weight_strategy: '{self.weight_strategy}'")

    ### Select Observation ###
    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, auxiliary_columns: list = None, **kwargs) -> dict:
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Calculate pairwise distances ##
        X_Candidate, _ = get_features_and_target(df=df_Candidate, target_column_name="Y", auxiliary_columns=auxiliary_columns)
        X_Train, y_Train = get_features_and_target(df=df_Train, target_column_name="Y", auxiliary_columns=auxiliary_columns)
    
        d_nmX = cdist(X_Candidate.values, X_Train.values, metric='euclidean')
        Predictions = Model.predict(X_Candidate)
        d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric='euclidean')

        ## Normalize distances ##
        epsilon = 1e-8
        d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        ## Weights ##
        w_x, w_y = 0.5, 0.5                         # Default weights
        if self.weight_strategy == 'static':
            w_x, w_y = self.w_x, self.w_y
        
        elif self.weight_strategy == 'time_decay':
            progress = self.iteration / self.initial_candidate_size
            
            if self.decay_type == 'linear':
                w_x = 1.0 - progress
            
            elif self.decay_type == 'exponential':
                w_x = np.exp(-self.decay_constant * progress)
            
            else:
                raise ValueError(f"Invalid decay_type: '{self.decay_type}'")

            w_y = 1.0 - w_x

        ## Final Score ##
        score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        final_scores = score_matrix.min(axis=1)
        best_candidate_iloc = np.argmax(final_scores)

        ## Update iteration ##
        self.iteration += 1

        ## Output ##
        IndexRecommendation = df_Candidate.iloc[[best_candidate_iloc]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)]}