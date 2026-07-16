### Import functions ###
import inspect
import numpy as np
import pandas as pd
import torch.nn as nn

### Functions ###
from utils.Selector import *
from utils.Prediction import *
from utils.Prediction.LightHydra import print_gpu_memory
from utils.Prediction.CrossValidation import get_cv_rmse

### Import functions ###
from sklearn.base import BaseEstimator
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerState
import time

import pickle


def compute_error(predictor_model, candidate_with_target, y_size, SimulationConfigInputUpdated_):

    SimulationConfigInputUpdated = SimulationConfigInputUpdated_

    StartTime = time.time()
    FullPoolErrorOuputs, SimulationConfigInputUpdated = FullPoolErrorFunction(
        InputModel=predictor_model,
        SimulationConfigInputUpdated=SimulationConfigInputUpdated,
        df_Candidate=candidate_with_target,
        y_size=y_size,
    )
    print(f"\t+++ FullPoolErrorOuputs : {time.time() - StartTime} +++")

    StartTime = time.time()
    FullTestErrorOuputs = FullTestErrorFunction(
        InputModel=predictor_model,
        SimulationConfigInputUpdated=SimulationConfigInputUpdated,
        df_test=SimulationConfigInputUpdated["df_test"],
        y_size=y_size,
    )
    print(f"\t+++ FullTestErrorOuputs : {time.time() - StartTime} +++")

    StartTime = time.time()
    TrainErrorOuputs = TrainErrorFunction(
        InputModel=predictor_model,
        SimulationConfigInputUpdated=SimulationConfigInputUpdated,
        df_train=SimulationConfigInputUpdated["df_Train"],
        y_size=y_size,
    )
    print(f"\t+++ TrainErrorOuputs : {time.time() - StartTime} +++")

    return (
        FullPoolErrorOuputs,
        FullTestErrorOuputs,
        TrainErrorOuputs,
        SimulationConfigInputUpdated,
    )


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
        "Train": {"RMSE": [], "MAE": [], "R2": [], "CC": []},
    }

    ErrorVecs_iteration = []
    WeightHistory = []
    SelectedObservationHistory = []
    InitialTrainIndices = list(SimulationConfigInputUpdated["df_Train"].index)

    y_size = SimulationConfigInputUpdated["y_size"]
    ### Initialize Model ###

    hydralightning = SimulationConfigInputUpdated["hl_trainer"] is not None

    if hydralightning:
        predictor_model = SimulationConfigInputUpdated["hl_trainer"]
        hl_model = SimulationConfigInputUpdated["hl_model"]
        hl_data = SimulationConfigInputUpdated["hl_data"]
        hl_cfg = SimulationConfigInputUpdated["hl_cfg"]

        data_train_full_len = len(SimulationConfigInputUpdated["df_full"])

        SimulationConfigInputUpdated["add_useful_params"]["k_top_candidate"] = min(
            data_train_full_len,
            SimulationConfigInputUpdated["add_useful_params"]["k_top_candidate"],
        )

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
        StartTime_iteration = time.time()

        print(f"=== iteration  {i} ===")

        ## 1. Get features and target for the current training set ##
        X_train_df, y_train_series = get_features_and_target(
            SimulationConfigInputUpdated["df_Train"], y_size=y_size
        )

        ## 2. Prediction Model ##
        if hydralightning:

            hl_data.train_data.update_indices(X_train_df.index)

            predictor_model, hl_model, hl_data, hl_cfg = update_scheduler(
                predictor_model, hl_model, hl_data, hl_cfg
            )

            reset_trainer(predictor_model)

            if not SimulationConfigInputUpdated["add_useful_params"]["curriculum"]:
                hl_model.apply(reset_weights)

            # hl_data_traindataloader = hl_data.train_dataloader()
            StartTime = time.time()

            print(f"\t*** Training Start ***")

            predictor_model.fit(model=hl_model, datamodule=hl_data, ckpt_path=None)

            print(f"\t+++ Training : {time.time() - StartTime} +++")

            SimulationConfigInputUpdated["hl_model"] = hl_model

        else:
            predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)

        ## 3. Calculate Full Pool Error ##
        candidate_with_target_index = SimulationConfigInputUpdated["df_Candidate"].index

        candidate_with_target = SimulationConfigInputUpdated["df_full"].loc[
            candidate_with_target_index, :
        ]

        if (
            i
            % SimulationConfigInputUpdated["add_useful_params"]["save_result_selection_frequency"]
            == 0
        ):

            (
                FullPoolErrorOuputs,
                FullTestErrorOuputs,
                TrainErrorOuputs,
                SimulationConfigInputUpdated,
            ) = compute_error(
                predictor_model, candidate_with_target, y_size, SimulationConfigInputUpdated
            )

            for metric_name, value in FullPoolErrorOuputs.items():
                ErrorVecs["Full_Pool"][metric_name].append(value)
            for metric_name, value in FullTestErrorOuputs.items():
                ErrorVecs["Full_Test"][metric_name].append(value)
            for metric_name, value in TrainErrorOuputs.items():
                ErrorVecs["Train"][metric_name].append(value)
            ErrorVecs_iteration.append(i)

        else:
            for metric_name, value in FullPoolErrorOuputs.items():
                ErrorVecs["Full_Pool"][metric_name].append(value)
            for metric_name, value in FullTestErrorOuputs.items():
                ErrorVecs["Full_Test"][metric_name].append(value)
            for metric_name, value in TrainErrorOuputs.items():
                ErrorVecs["Train"][metric_name].append(value)

            ErrorVecs_iteration.append(ErrorVecs_iteration[-1])

        ## 4. Calculate CV Error ##
        model = predictor_model.model

        StartTime = time.time()

        if model is not None:

            if isinstance(model, BaseEstimator):
                current_cv_rmse = (
                    0
                    if SimulationConfigInputUpdated["add_useful_params"]["no_cv"]
                    else get_cv_rmse(model, X_train_df, y_train_series, k=5)
                )
            # elif isinstance(model, nn.Module):
            #     current_cv_rmse = get_cv_rmse_NN(model, X_train_df, y_train_series, k=5)
            elif hydralightning:
                current_cv_rmse = (
                    0
                    if SimulationConfigInputUpdated["add_useful_params"]["no_cv"]
                    else get_cv_rmse_hl(predictor_model, model, hl_data, hl_cfg)
                )

                # current_cv_rmse = get_cv_rmse_hl_multigpu(hl_cfg)

            else:
                raise "current_cv_rmse class is not known"

        else:
            current_cv_rmse = np.nan

        print(f"\t+++ COMPUTE CV : {time.time() - StartTime} +++")

        if np.isnan(current_cv_rmse):
            raise "current_cv_rmse is Nan"
            # current_cv_rmse = FullPoolErrorOuputs["RMSE"]

        ### 5. Break Condition ###
        if len(SimulationConfigInputUpdated["df_Candidate"]) == 0:
            (
                FullPoolErrorOuputs,
                FullTestErrorOuputs,
                TrainErrorOuputs,
                SimulationConfigInputUpdated,
            ) = compute_error(
                predictor_model, candidate_with_target, y_size, SimulationConfigInputUpdated
            )
            for metric_name, value in FullPoolErrorOuputs.items():
                ErrorVecs["Full_Pool"][metric_name][-1] = value
            for metric_name, value in FullTestErrorOuputs.items():
                ErrorVecs["Full_Test"][metric_name][-1] = value
            for metric_name, value in TrainErrorOuputs.items():
                ErrorVecs["Train"][metric_name][-1] = value
            ErrorVecs_iteration[-1] = i

            break

        StartTime = time.time()

        ## 6. Sampling Procedure ##
        SelectorFuncOutput = selector_model.select(
            df_Candidate=SimulationConfigInputUpdated["df_Candidate"],
            df_Train=SimulationConfigInputUpdated["df_Train"],
            y_size=y_size,
            Model=predictor_model,
            current_rmse=current_cv_rmse,
            SimulationConfigInputUpdated=SimulationConfigInputUpdated,
        )
        print(f"\t+++ SelectorFuncOutput : {time.time() - StartTime} +++")

        StartTime = time.time()

        ## 7. Query selected observation ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]

        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]

        SelectedObservationHistory.append(QueryObservationIndex)

        ## 8. Store weights ##
        w_x = SelectorFuncOutput.get("w_x", np.nan)
        WeightHistory.append(w_x)

        # load target 'Y' only for train and not for candidate

        # Work with TRUE INDICES ! .iloc -> rempace by .loc
        # QueryObservation = SimulationConfigInputUpdated["df_full"].iloc[QueryObservationIndex, :]
        QueryObservation = SimulationConfigInputUpdated["df_full"].loc[QueryObservationIndex, :]

        ## 9. Update Train and Candidate Sets ##
        SimulationConfigInputUpdated["df_Train"] = pd.concat(
            [SimulationConfigInputUpdated["df_Train"], QueryObservation]
        )

        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated[
            "df_Candidate"
        ].drop(QueryObservationIndex)

        ## 10. Increase iteration ##
        i += 1

        print(f"\t+++ #7-8-7-10 LearningProcedure : {time.time() - StartTime} +++")

        ## 11. Increase iteration ##
        StartTime = time.time()
        dump_results(
            SimulationConfigInputUpdated,
            ErrorVecs,
            SelectedObservationHistory,
            WeightHistory,
            InitialTrainIndices,
            ErrorVecs_iteration,
        )
        output_path = SimulationConfigInputUpdated["add_useful_params"]["output_path"]
        print(f"\n=== Results writen in {output_path} ===\n")
        print(f"\t+++ dump_results : {time.time() - StartTime} +++")

        print(f"\t+++ ITERATION {i} : {time.time() - StartTime_iteration} +++")

    ### Output ###
    LearningProcedureOutput = {
        "ErrorVecs": ErrorVecs,
        "SelectedObservationHistory": SelectedObservationHistory,
        "WeightHistory": WeightHistory,
        "InitialTrainIndices": InitialTrainIndices,
        "ErrorVecs_iteration": ErrorVecs_iteration,
    }
    return LearningProcedureOutput


def dump_results(
    SimulationConfigInputUpdated,
    ErrorVecs,
    SelectedObservationHistory,
    WeightHistory,
    InitialTrainIndices,
    ErrorVecs_iteration,
):

    ### Output ###
    LearningProcedureOutput = {
        "ErrorVecs": ErrorVecs,
        "SelectedObservationHistory": SelectedObservationHistory,
        "WeightHistory": WeightHistory,
        "InitialTrainIndices": InitialTrainIndices,
        "ErrorVecs_iteration": ErrorVecs_iteration,
    }
    # return LearningProcedureOutput

    ### Return Simulation Parameters ###
    SimulationParameters = {
        "DataFileInput": str(SimulationConfigInputUpdated["DataFileInput"]),
        "Seed": str(SimulationConfigInputUpdated["Seed"]),
        "CandidateProportion": str(SimulationConfigInputUpdated["CandidateProportion"]),
        "SelectorType": str(SimulationConfigInputUpdated["SelectorType"]),
        "ModelType": str(SimulationConfigInputUpdated["ModelType"]),
        "output_path": str(SimulationConfigInputUpdated["add_useful_params"]["output_path"]),
        "save_result_selection_frequency": str(
            SimulationConfigInputUpdated["add_useful_params"]["save_result_selection_frequency"]
        ),
        "k_top_candidate": str(
            SimulationConfigInputUpdated["add_useful_params"]["k_top_candidate"]
        ),
        "subset_rand_candidat": str(
            SimulationConfigInputUpdated["add_useful_params"]["subset_rand_candidat"]
        ),
        "hl_xp": str(SimulationConfigInputUpdated["add_useful_params"]["hl_xp"]),
        "strat": str(SimulationConfigInputUpdated["add_useful_params"]["strat"]),
        "no_cv": str(SimulationConfigInputUpdated["add_useful_params"]["no_cv"]),
        "hl_max_epoch": str(SimulationConfigInputUpdated["add_useful_params"]["hl_max_epoch"]),
        "curriculum": str(SimulationConfigInputUpdated["add_useful_params"]["curriculum"]),
    }

    ### Return Time ###
    ElapsedTime = time.time() - SimulationConfigInputUpdated["StartTime"]

    ### Return Dictionary ###
    ErrorVecs = pd.DataFrame(LearningProcedureOutput["ErrorVecs"])

    SimulationResults = {
        "ErrorVecs": ErrorVecs,
        "SelectionHistory": LearningProcedureOutput["SelectedObservationHistory"],
        "WeightHistory": LearningProcedureOutput["WeightHistory"],
        "InitialTrainIndices": LearningProcedureOutput["InitialTrainIndices"],
        "SimulationParameters": SimulationParameters,
        "ElapsedTime": ElapsedTime,
        "k_top_candidate": SimulationConfigInputUpdated["add_useful_params"]["k_top_candidate"],
        "ErrorVecs_iteration": LearningProcedureOutput["ErrorVecs_iteration"],
    }

    print("SimulationResults", SimulationResults.keys())
    if "df_full" in SimulationConfigInputUpdated:
        SimulationResults["TotalPoolSize"] = len(SimulationConfigInputUpdated["df_full"])

    results = SimulationResults

    all_results_by_strategy = {}

    strategy_name = SimulationConfigInputUpdated["add_useful_params"]["strategy_name"]

    # Store results #
    all_results_by_strategy[strategy_name] = results
    print("results", results.keys())

    ### Run the Simulation for a Single Seed and Model ###
    SimulationResults = all_results_by_strategy

    output_path = SimulationConfigInputUpdated["add_useful_params"]["output_path"]

    ### Save Simulation Results to the new nested directory ###
    with open(output_path, "wb") as f:
        pickle.dump(SimulationResults, f)
