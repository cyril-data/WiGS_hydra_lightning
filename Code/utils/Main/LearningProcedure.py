### Import functions ###
import inspect
import numpy as np
import pandas as pd
import torch.nn as nn

### Functions ###
from utils.Selector import *
from utils.Prediction import *
from utils.Prediction.CrossValidation import get_cv_rmse

### Import functions ###
from sklearn.base import BaseEstimator


### Function ###
def LearningProcedure(SimulationConfigInputUpdated):
    """
    Executes an iterative active learning or greedy sampling loop.

    Args:
        SimulationConfigInputUpdated (dict): A dictionary containing the configuration and state for the learning loop.

    Returns:
        dict: A dictionary containing the results of the learning procedure with the following keys:
            - ErrorVec (dict): A dictionary (with key 'Full_Pool') where values are dictionaries
              of metric names ('RMSE', 'MAE', 'R2', 'CC') and lists of the metric's value at each iteration.
            - SelectedObservationHistory (list): A list of the indices of observations selected from the candidate pool
              in the order they were chosen."""

    ### Set Up ###
    i = 0
    ErrorVecs = {
        "Full_Pool": {"RMSE": [], "MAE": [], "R2": [], "CC": []},
        "Full_Test": {"RMSE": [], "MAE": [], "R2": [], "CC": []},
    }
    WeightHistory = []
    SelectedObservationHistory = []
    InitialTrainIndices = list(SimulationConfigInputUpdated["df_Train"].index)
    y_size = SimulationConfigInputUpdated["y_size"]
    ### Initialize Model ###

    if SimulationConfigInputUpdated["hl_trainer"] is not None:
        predictor_model = SimulationConfigInputUpdated["hl_trainer"]
    else:
        ModelClass = globals().get(SimulationConfigInputUpdated["ModelType"], None)
        model_init_args = {
            k: v
            for k, v in SimulationConfigInputUpdated.items()
            if k in inspect.signature(ModelClass.__init__).parameters
        }
        predictor_model = ModelClass(**model_init_args)
        SimulationConfigInputUpdated["Model"] = predictor_model

    ### Initialize Selector ###
    if "df_Candidate" in SimulationConfigInputUpdated:
        SimulationConfigInputUpdated["initial_candidate_size"] = len(
            SimulationConfigInputUpdated["df_Candidate"]
        )
    SelectorClass = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
    selector_init_args = {
        k: v
        for k, v in SimulationConfigInputUpdated.items()
        if k in inspect.signature(SelectorClass.__init__).parameters
    }
    selector_model = SelectorClass(**selector_init_args)

    ### Algorithm ###
    while True:

        ## 1. Get features and target for the current training set ##
        X_train_df, y_train_series = get_features_and_target(
            SimulationConfigInputUpdated["df_Train"], "Y"
        )

        ## 2. Prediction Model ##
        if SimulationConfigInputUpdated["hl_trainer"] is not None:
            predictor_model.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        else:
            predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)
        ## 3. Calculate Full Pool Error ##
        candidate_with_target_index = SimulationConfigInputUpdated["df_Candidate"].index

        # TODO : multivariate
        # load target 'Y' only for train and not for candidate

        candidate_with_target = SimulationConfigInputUpdated["df_full"].iloc[
            candidate_with_target_index, :
        ]

        FullPoolErrorOuputs = FullPoolErrorFunction(
            InputModel=predictor_model,
            df_Train=SimulationConfigInputUpdated["df_Train"],
            df_Candidate=candidate_with_target,
            # y_size=y_size,
        )
        for metric_name, value in FullPoolErrorOuputs.items():
            ErrorVecs["Full_Pool"][metric_name].append(value)

        FullTestErrorOuputs = FullTestErrorFunction(
            InputModel=predictor_model,
            df_test=SimulationConfigInputUpdated["df_test"],
            # y_size=y_size,
        )
        for metric_name, value in FullTestErrorOuputs.items():
            ErrorVecs["Full_Test"][metric_name].append(value)

        ## 4. Calculate CV Error ##
        model = predictor_model.model

        if model is not None:
            if isinstance(model, BaseEstimator):
                current_cv_rmse = get_cv_rmse(model, X_train_df, y_train_series, k=5)
            elif isinstance(model, nn.Module):
                current_cv_rmse = get_cv_rmse_NN(model, X_train_df, y_train_series, k=5)
            else:
                raise "current_cv_rmse class is not known"
        else:
            current_cv_rmse = np.nan
        if np.isnan(current_cv_rmse):
            raise "current_cv_rmse is Nan"
            # current_cv_rmse = FullPoolErrorOuputs["RMSE"]

        ### 5. Break Condition ###
        if len(SimulationConfigInputUpdated["df_Candidate"]) == 0:
            break

        ## 6. Sampling Procedure ##
        SelectorFuncOutput = selector_model.select(
            df_Candidate=SimulationConfigInputUpdated["df_Candidate"],
            df_Train=SimulationConfigInputUpdated["df_Train"],
            y_size=y_size,
            Model=predictor_model,
            current_rmse=current_cv_rmse,
        )

        ## 7. Query selected observation ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex)

        ## 8. Store weights ##
        w_x = SelectorFuncOutput.get("w_x", np.nan)
        WeightHistory.append(w_x)

        # load target 'Y' only for train and not for candidate
        # TODO : multivariate
        QueryObservation = SimulationConfigInputUpdated["df_full"].iloc[QueryObservationIndex, :]

        ## 9. Update Train and Candidate Sets ##
        SimulationConfigInputUpdated["df_Train"] = pd.concat(
            [SimulationConfigInputUpdated["df_Train"], QueryObservation]
        )

        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated[
            "df_Candidate"
        ].drop(QueryObservationIndex)

        ## 10. Increase iteration ##
        i += 1

    ### Output ###
    LearningProcedureOutput = {
        "ErrorVecs": ErrorVecs,
        "SelectedObservationHistory": SelectedObservationHistory,
        "WeightHistory": WeightHistory,
        "InitialTrainIndices": InitialTrainIndices,
    }
    return LearningProcedureOutput
