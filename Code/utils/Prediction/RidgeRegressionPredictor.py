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
        self.X_train_cached = None 

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.model = Ridge(alpha=self.regularization)
        self.model.fit(X_train_df, y_train_series)
        

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)
    
    ### Predict with Uncertainty ###
    def predict_with_std(self, X_data_df: pd.DataFrame):
        """
        Calculates the standard deviation of prediction for Ridge Regression.
        Std[x] = sqrt( x^T * (X_train^T * X_train + alpha * I)^-1 * x )
        """
        X_cand = X_data_df.values
        n_features = self.X_train_cached.shape[1]
        
        # 1. Compute Inverse Covariance (Precision Matrix)
        XTX = np.dot(self.X_train_cached.T, self.X_train_cached)
        regularizer = self.regularization * np.eye(n_features)
        
        try:
            inv_covariance = np.linalg.inv(XTX + regularizer)
        except np.linalg.LinAlgError:
            inv_covariance = np.linalg.pinv(XTX + regularizer)
            
        # 2. Compute Variance efficiently
        intermediate = np.dot(X_cand, inv_covariance)
        variances = np.sum(intermediate * X_cand, axis=1)
        
        # 3. Return Mean and Std
        mean = self.model.predict(X_data_df)
        std = np.sqrt(np.maximum(variances, 0))
        
        return mean, std