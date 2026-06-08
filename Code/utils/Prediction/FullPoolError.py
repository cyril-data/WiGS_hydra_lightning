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
    _, y_true_pool = get_features_and_target(df_pool, target_column_name="Y")

    # 2. Get features and labels from the separate sets.
    _, y_train = get_features_and_target(df_Train, target_column_name="Y")

    # 3. Check if the candidate pool is empty
    if df_Candidate.empty:  # On the last loop, the hybrid vector is just the training labels
        y_hybrid_predictions = y_train
    else:  # Otherwise, predict on the candidate set

        X_candidate, _ = get_features_and_target(df_Candidate, target_column_name="Y")

        # Check again in case X_candidate is empty but df_Candidate was not
        if X_candidate.empty:
            y_hybrid_predictions = y_train
        else:

            y_pred_candidate = InputModel.predict(X_candidate)

            # y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_candidate.index)

            y_pred_candidate_pd = pd.DataFrame(
                y_pred_candidate,
                index=X_candidate.index,
                columns=y_train.columns.to_list(),  # Nom de la colonne
            )

            # 4. Construct the hybrid prediction vector.
            y_hybrid_predictions = pd.concat([y_train, y_pred_candidate_pd])

    # 5. Ensure the final vectors are aligned by index.
    y_hybrid_predictions = y_hybrid_predictions.loc[y_true_pool.index]

    # 6. Calculate all metrics using the same logic for every iteration.
    rmse = np.sqrt(mean_squared_error(y_true_pool, y_hybrid_predictions))
    mae = mean_absolute_error(y_true_pool, y_hybrid_predictions)
    r2 = r2_score(y_true_pool, y_hybrid_predictions)

    correlations = []

    # Pour chaque colonne, calculer la corrélation
    for col in y_true_pool.columns:
        if np.std(y_true_pool[col]) > 0 and np.std(y_hybrid_predictions[col]) > 0:
            corr = np.corrcoef(y_true_pool[col], y_hybrid_predictions[col])[0, 1]
        else:
            corr = 1.0
        correlations.append(corr)

    # Moyenne des corrélations
    cc = np.mean(correlations)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CC": cc}


### Function ###
def FullTestErrorFunction(InputModel, df_test: pd.DataFrame) -> dict:
    """
    Calculates performance metrics using the hybrid evaluation method from Wu et al. (2018).

    This method evaluates performance on the Test set.
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
    df_pool = df_test
    X_true_pool, y_true_pool = get_features_and_target(df_pool, target_column_name="Y")

    # Check again in case X_candidate is empty but df_Candidate was not
    y_pred_candidate = InputModel.predict(X_true_pool)
    y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_true_pool.index)

    # 6. Calculate all metrics using the same logic for every iteration.
    rmse = np.sqrt(mean_squared_error(y_true_pool, y_pred_candidate))
    mae = mean_absolute_error(y_true_pool, y_pred_candidate)
    r2 = r2_score(y_true_pool, y_pred_candidate)

    y_pred_candidate_pd = pd.DataFrame(
        y_pred_candidate,
        columns=y_true_pool.columns.to_list(),  # Nom de la colonne
    )

    correlations = []

    # Pour chaque colonne, calculer la corrélation
    for col in y_true_pool.columns:
        if np.std(y_true_pool[col]) > 0 and np.std(y_pred_candidate_pd[col]) > 0:
            corr = np.corrcoef(y_true_pool[col], y_pred_candidate_pd[col])[0, 1]
        else:
            corr = 1.0
        correlations.append(corr)

    # Moyenne des corrélations
    cc = np.mean(correlations)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CC": cc}


# ### Function ###
# def FullPoolErrorFunction(
#     InputModel, df_Train: pd.DataFrame, df_Candidate: pd.DataFrame, y_size: int
# ) -> dict:
#     """
#     Calculates performance metrics using the hybrid evaluation method from Wu et al. (2018).

#     This method evaluates performance on the entire data pool (training + candidate).
#     It uses the true labels for the training set and the model's predictions for the
#     candidate set to form a "hybrid" prediction vector, which is then compared against
#     the true labels of the entire pool.

#     Args:
#         InputModel (object): A trained model object with a .predict() method.
#         df_Train (pd.DataFrame): The current training dataset.
#         df_Candidate (pd.DataFrame): The current candidate dataset.

#     Returns:
#         dict: A dictionary containing the calculated metrics: 'RMSE', 'MAE', 'R2', and 'CC'.
#     """
#     # 1. Recreate the full data pool.
#     df_pool = pd.concat([df_Train, df_Candidate])
#     _, y_true_pool = get_features_and_target(df_pool, y_size)

#     print("y_true_pool IS NAN", y_true_pool.isnull().sum())

#     # 2. Get features and labels from the separate sets.
#     _, y_train = get_features_and_target(df_Train, y_size)

#     print("y_train IS NAN", y_train.isnull().sum())
#     # 3. Check if the candidate pool is empty
#     if df_Candidate.empty:  # On the last loop, the hybrid vector is just the training labels
#         y_hybrid_predictions = y_train
#     else:  # Otherwise, predict on the candidate set
#         X_candidate, _ = get_features_and_target(df_Candidate, y_size)

#         # Check again in case X_candidate is empty but df_Candidate was not
#         if X_candidate.empty:
#             y_hybrid_predictions = y_train
#         else:
#             y_pred_candidate = InputModel.predict(X_candidate)
#             y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_candidate.index)

#             # 4. Construct the hybrid prediction vector.
#             y_hybrid_predictions = pd.concat([y_train, y_pred_candidate_series])

#     print("y_hybrid_predictions IS NAN", y_hybrid_predictions.isnull().sum())

#     print("y_true_pool.index", len(y_true_pool.index))
#     print("y_hybrid_predictions.index", len(y_hybrid_predictions.index))
#     # 5. Ensure the final vectors are aligned by index.
#     y_hybrid_predictions = y_hybrid_predictions.loc[y_true_pool.index]

#     # 6. Calculate all metrics using the same logic for every iteration.
#     print("y_true_pool", y_true_pool.isnull().sum())
#     print("y_hybrid_predictions", y_hybrid_predictions.isnull().sum())

#     rmse = np.sqrt(mean_squared_error(y_true_pool, y_hybrid_predictions))
#     mae = mean_absolute_error(y_true_pool, y_hybrid_predictions)
#     r2 = r2_score(y_true_pool, y_hybrid_predictions)

#     # Handle the zero-variance edge case for the correlation coefficient.
#     if np.std(y_hybrid_predictions) > 0 and np.std(y_true_pool) > 0:
#         cc = np.corrcoef(y_true_pool, y_hybrid_predictions)[0, 1]
#     else:
#         cc = 1.0

#     return {"RMSE": rmse, "MAE": mae, "R2": r2, "CC": cc}


# ### Function ###
# def FullTestErrorFunction(InputModel, df_test: pd.DataFrame, y_size: int) -> dict:
#     """
#     Calculates performance metrics using the hybrid evaluation method from Wu et al. (2018).

#     This method evaluates performance on the Test set.
#     It uses the true labels for the training set and the model's predictions for the
#     candidate set to form a "hybrid" prediction vector, which is then compared against
#     the true labels of the entire pool.

#     Args:
#         InputModel (object): A trained model object with a .predict() method.
#         df_Train (pd.DataFrame): The current training dataset.
#         df_Candidate (pd.DataFrame): The current candidate dataset.

#     Returns:
#         dict: A dictionary containing the calculated metrics: 'RMSE', 'MAE', 'R2', and 'CC'.
#     """
#     # 1. Recreate the full data pool.
#     df_pool = df_test
#     X_true_pool, y_true_pool = get_features_and_target(df_pool, y_size)

#     # Check again in case X_candidate is empty but df_Candidate was not
#     y_pred_candidate = InputModel.predict(X_true_pool)
#     y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_true_pool.index)

#     # 6. Calculate all metrics using the same logic for every iteration.
#     rmse = np.sqrt(mean_squared_error(y_true_pool, y_pred_candidate))
#     mae = mean_absolute_error(y_true_pool, y_pred_candidate)
#     r2 = r2_score(y_true_pool, y_pred_candidate)

#     # Handle the zero-variance edge case for the correlation coefficient.
#     if np.std(y_pred_candidate) > 0 and np.std(y_true_pool) > 0:
#         cc = np.corrcoef(y_true_pool, y_pred_candidate)[0, 1]
#     else:
#         cc = 1.0

#     return {"RMSE": rmse, "MAE": mae, "R2": r2, "CC": cc}
