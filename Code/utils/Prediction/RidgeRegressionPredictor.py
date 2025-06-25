# utils/Prediction/RidgeRegressionPredictor.py

# Summary: Initializes and fits a ridge regression model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   regularization: Ridge regression penalty (alpha_val is mapped to regularization).
# Output:
# RidgeRegressionModel: A ridge regression model instance.

### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.linear_model import Ridge

class RidgeRegressionPredictor:

    ### Initialize Model ###
    def __init__(self, regularization: float = 1.0, **kwargs):
        self.regularization = regularization
        self.model = None 

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.model = Ridge(alpha=self.regularization)
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)