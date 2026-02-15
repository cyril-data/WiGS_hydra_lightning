### Libraries ###
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestRegressorPredictor:
    """
    A wrapper for the scikit-learn RandomForestRegressor model.
    """

    ### Initialize Model ###
    def __init__(self, Seed: int, n_estimators: int = 100, **kwargs):
        self.Seed = Seed
        self.n_estimators = n_estimators
        self.model = None 
        np.random.seed(self.Seed)

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        # We re-initialize the model each time to ensure a fresh start
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.Seed,
            n_jobs=-1 
        )
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)

    ### Predict with Uncertainty (The Magic Trick) ###
    def predict_with_std(self, X_data_df: pd.DataFrame):
        """
        Calculates uncertainty by looking at the spread of predictions 
        across the individual trees in the forest.
        
        Returns:
            (mean, std) tuple
        """
        # 1. Get predictions from every single tree
        individual_preds = np.array([
            tree.predict(X_data_df.values) 
            for tree in self.model.estimators_
        ])
        
        # 2. Calculate Mean (Standard Prediction)
        mean_pred = np.mean(individual_preds, axis=0)
        
        # 3. Calculate Standard Deviation (Uncertainty)
        std_pred = np.std(individual_preds, axis=0)
        
        return mean_pred, std_pred