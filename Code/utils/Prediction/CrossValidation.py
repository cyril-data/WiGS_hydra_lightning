import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

def get_cv_rmse(model_object, X_train, y_train, k=5):
    """
    Calculates the K-fold CV RMSE for the current training set.

    Args:
        model_object: The scikit-learn compatible model object 
                      (e.g., the .model inside your RidgeRegressionPredictor).
        X_train (pd.DataFrame): Current training features.
        y_train (pd.Series): Current training labels.
        k (int): Number of folds.
    """

    ### Set up ###
    if len(y_train) < k * 2: 
        return np.nan 
    model_to_cv = model_object

    ### Define the scoring ###
    scores = cross_val_score(model_to_cv, 
                             X_train, 
                             y_train, 
                             cv=KFold(n_splits=k, shuffle=True, random_state=42), 
                             scoring='neg_root_mean_squared_error')
    return -np.mean(scores)