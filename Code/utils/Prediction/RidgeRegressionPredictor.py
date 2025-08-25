### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.linear_model import Ridge

class RidgeRegressionPredictor:
    """
    A wrapper for the scikit-learn Ridge model.
    """
    
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