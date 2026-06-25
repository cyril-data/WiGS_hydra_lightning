### Import libraries ###
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target
import torch
from collections import defaultdict
from utils.Prediction.LightHydra import hl_pd_to_dataloader, hl_y_pred_pd_to_tensor


### Function ###
def FullPoolErrorFunction(
    InputModel, SimulationConfigInputUpdated: dict, df_Candidate: pd.DataFrame, y_size=1
) -> dict:
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

    df_Train = SimulationConfigInputUpdated["df_Train"]

    df_pool = pd.concat([df_Train, df_Candidate])
    _, y_true_pool = get_features_and_target(df_pool, y_size=y_size)

    # 2. Get features and labels from the separate sets.
    _, y_train = get_features_and_target(df_Train, y_size=y_size)

    hydralightning = SimulationConfigInputUpdated["hl_trainer"] is not None

    print("FullPoolErrorFunction df_Train", df_Train.shape)
    print("FullPoolErrorFunction y_true_pool", y_true_pool.shape)

    # 3. Check if the candidate pool is empty
    if df_Candidate.empty:  # On the last loop, the hybrid vector is just the training labels
        y_hybrid_predictions = y_train
        if hydralightning and "all_reg_cols" in SimulationConfigInputUpdated:
            all_reg_cols = SimulationConfigInputUpdated["all_reg_cols"]
            y_hybrid_predictions = y_hybrid_predictions[all_reg_cols]

    else:  # Otherwise, predict on the candidate set

        X_candidate, _ = get_features_and_target(df_Candidate, y_size=y_size)

        # Check again in case X_candidate is empty but df_Candidate was not
        if X_candidate.empty:
            y_hybrid_predictions = y_train
            if hydralightning and "all_reg_cols" in SimulationConfigInputUpdated:
                all_reg_cols = SimulationConfigInputUpdated["all_reg_cols"]
                y_hybrid_predictions = y_hybrid_predictions[all_reg_cols]
        else:
            # trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
            if hydralightning:

                hl_model = SimulationConfigInputUpdated["hl_model"]
                hl_data = SimulationConfigInputUpdated["hl_data"]

                # dataloader = hl_pd_to_dataloader(X_candidate, hl_model.device, hl_model.dtype)

                # Après (réutilise le même DataLoader)
                candidate_indices = X_candidate.index.tolist()
                hl_data.pred_data.update_indices(candidate_indices)

                y_pred_candidate = InputModel.predict(model=hl_model, datamodule=hl_data)
                # y_pred_candidate = InputModel.predict(model=hl_model, dataloaders=dataloader)

                y_pred_candidate_pd, all_reg_cols = hl_y_pred_pd_to_tensor(
                    y_pred_candidate, y_train.columns.to_list(), X_candidate.index
                )

                if "all_reg_cols" not in SimulationConfigInputUpdated:
                    SimulationConfigInputUpdated["all_reg_cols"] = all_reg_cols

                # restrain only on regression
                y_train = y_train[all_reg_cols]

                # 4. Construct the hybrid prediction vector.
                y_hybrid_predictions = pd.concat([y_train, y_pred_candidate_pd])

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

    if hydralightning:
        y_true_pool = y_true_pool[all_reg_cols]

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

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CC": cc}, SimulationConfigInputUpdated


### Function ###
def FullTestErrorFunction(
    InputModel, SimulationConfigInputUpdated: dict, df_test: pd.DataFrame, y_size: int = 1
) -> dict:
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
    hydralightning = SimulationConfigInputUpdated["hl_trainer"] is not None

    # 1. Recreate the full data pool.

    X_true_pool, y_true_pool = get_features_and_target(df_test, y_size=y_size)

    print("FullTestErrorFunction y_true_pool", y_true_pool.shape)

    if hydralightning:
        hl_model = SimulationConfigInputUpdated["hl_model"]
        hl_data = SimulationConfigInputUpdated["hl_data"]

        # Après (réutilise le même DataLoader)
        candidate_indices = X_true_pool.index.tolist()

        hl_data.pred_data.update_indices(candidate_indices)

        # batch_size = hl_data.batch_size_per_device
        # hl_data.batch_size_per_device = batch_size * 10

        y_pred_candidate = InputModel.predict(model=hl_model, datamodule=hl_data)

        # hl_data.batch_size_per_device = batch_size

        y_pred_candidate_pd, all_reg_cols = hl_y_pred_pd_to_tensor(
            y_pred_candidate, y_true_pool.columns.to_list(), X_true_pool.index
        )

        y_pred_candidate = y_pred_candidate_pd

        # restrain only on regression
        y_true_pool = y_true_pool[all_reg_cols]

    else:

        # Check again in case X_candidate is empty but df_Candidate was not
        y_pred_candidate = InputModel.predict(X_true_pool)
        y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_true_pool.index)

        y_pred_candidate_pd = pd.DataFrame(
            y_pred_candidate,
            columns=y_true_pool.columns.to_list(),  # Nom de la colonne
        )

    # 6. Calculate all metrics using the same logic for every iteration.
    rmse = np.sqrt(mean_squared_error(y_true_pool, y_pred_candidate))
    mae = mean_absolute_error(y_true_pool, y_pred_candidate)
    r2 = r2_score(y_true_pool, y_pred_candidate)

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


### Function ###
def TrainErrorFunction(
    InputModel, SimulationConfigInputUpdated: dict, df_train: pd.DataFrame, y_size: int = 1
) -> dict:
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
    hydralightning = SimulationConfigInputUpdated["hl_trainer"] is not None

    # 1. Recreate the full data pool.

    X_true_pool, y_true_pool = get_features_and_target(df_train, y_size=y_size)

    print("TrainErrorFunction y_true_pool", y_true_pool.shape)

    if hydralightning:
        hl_model = SimulationConfigInputUpdated["hl_model"]
        hl_data = SimulationConfigInputUpdated["hl_data"]

        # Après (réutilise le même DataLoader)
        candidate_indices = X_true_pool.index.tolist()

        hl_data.pred_data.update_indices(candidate_indices)

        y_pred_candidate = InputModel.predict(model=hl_model, datamodule=hl_data)

        y_pred_candidate_pd, all_reg_cols = hl_y_pred_pd_to_tensor(
            y_pred_candidate, y_true_pool.columns.to_list(), X_true_pool.index
        )

        y_pred_candidate = y_pred_candidate_pd

        # restrain only on regression
        y_true_pool = y_true_pool[all_reg_cols]

    else:

        # Check again in case X_candidate is empty but df_Candidate was not
        y_pred_candidate = InputModel.predict(X_true_pool)
        y_pred_candidate_series = pd.Series(y_pred_candidate, index=X_true_pool.index)

        y_pred_candidate_pd = pd.DataFrame(
            y_pred_candidate,
            columns=y_true_pool.columns.to_list(),  # Nom de la colonne
        )

    # 6. Calculate all metrics using the same logic for every iteration.
    rmse = np.sqrt(mean_squared_error(y_true_pool, y_pred_candidate))
    mae = mean_absolute_error(y_true_pool, y_pred_candidate)
    r2 = r2_score(y_true_pool, y_pred_candidate)

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
