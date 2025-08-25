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
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.Seed)
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)