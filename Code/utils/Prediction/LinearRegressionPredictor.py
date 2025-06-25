# Summary: Initializes and fits a linear regression model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
# Output:
# LinearRegressionModel: A linear regression model instance.

### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

class LinearRegressionPredictor:

    ### Initialize Model ###
    def __init__(self, **kwargs):
        self.model = None

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.model = LinearRegression()
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)