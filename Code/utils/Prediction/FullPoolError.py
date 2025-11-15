### Import libraries ###
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
def FullPoolErrorFunction(InputModel, df_Train: pd.DataFrame, df_Candidate: pd.DataFrame) -> dict:
    """
    Calculates performance metrics using the hybrid evaluation method from Wu et al. (2018).

    This method evaluates performance on the entire data pool (training + candidate).
    It uses the true labels for the training set and the model's predictions for the
    candidate set to form a "hybrid" prediction vector, which is then compared against
    the true labels of the entire pool.

    Args:
        InputModel (object): A trained model object with a .predict() method.
        df_Train (pd.DataFrame): The current training dataset.
        df_Candidate (pd.DataFrame): The current candidate dataset.

    Returns:
        dict: A dictionary containing the calculated metrics: 'RMSE', 'MAE', 'R2', and 'CC'.
    """
    # 1. Recreate the full data pool. 
    df_pool = pd.concat([df_Train, df_Candidate])
    _, y_true_pool = get_features_and_target(df_pool, "Y")

    # 2. Get features and labels from the separate sets.
    _, y_train = get_features_and_target(df_Train, "Y")

    # 3. Check if the candidate pool is empty (i.e., final iteration)
    if df_Candidate.empty:                  # On the last loop, the hybrid vector is just the training labels
        y_hybrid_predictions = y_train
    else:                                   # Otherwise, predict on the candidate set
        X_candidate, _ = get_features_and_target(df_Candidate, "Y")
        
        # Check again in case X_candidate is empty but df_Candidate was not
        if X_candidate.empty:
            y_hybrid_predictions = y_train
        else:
            y_pred_candidate = InputModel.predict(X_candidate)
            y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_candidate.index)

            # 4. Construct the hybrid prediction vector.
            y_hybrid_predictions = pd.concat([y_train, y_pred_candidate_series])

    # 5. Ensure the final vectors are aligned by index.
    y_hybrid_predictions = y_hybrid_predictions.loc[y_true_pool.index]

    # 6. Calculate all metrics using the same logic for every iteration.
    rmse = np.sqrt(mean_squared_error(y_true_pool, y_hybrid_predictions))
    mae = mean_absolute_error(y_true_pool, y_hybrid_predictions)
    r2 = r2_score(y_true_pool, y_hybrid_predictions)

    # Handle the zero-variance edge case for the correlation coefficient.
    if np.std(y_hybrid_predictions) > 0 and np.std(y_true_pool) > 0:
        cc = np.corrcoef(y_true_pool, y_hybrid_predictions)[0, 1]
    else:
        cc = 1.0

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CC': cc}