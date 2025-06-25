# Summary: Initializes and fits a random forest regressor model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   n_estimators: The number of trees for a random forest.
#   Seed: Seed for reproducibility.
# Output:
# RandomForestRegressorModel: A random forest regressor model instance.

### Libraries ###
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestRegressorPredictor:

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