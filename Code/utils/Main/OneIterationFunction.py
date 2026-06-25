### Import packages ###
import time
import numpy as np
import pandas as pd
import random as random

### Import functions ###
from utils.Auxiliary import (
    LoadData,
    get_target_rows,
    get_target_columns,
    get_target_columns_except_first,
    read_safetensor,
)
from utils.Main.LearningProcedure import LearningProcedure
from utils.Main.TrainCandidateSplit import (
    TrainCandidateSplit,
    TrainCandidateSplit_X,
)
from utils.Prediction.LightHydra import (
    get_hl_cfg,
    get_hl_modules,
    get_hl_datamodules,
    full_datamodule_to_pd,
    update_scheduler,
)
import os

### Isolate candidate target into function


### Function ###
def OneIterationFunction(SimulationConfigInput):
    """
    Executes a single, complete simulation iteration for a given configuration.

    Args:
        SimulationConfigInput (dict): A dictionary containing all parameters needed to run the simulation.

    Returns:
        dict: A dictionary containing the results of the simulation run, with the following keys:
            - ErrorVecs (pd.DataFrame): The history of performance metrics over the course of the learning procedure.
            - SelectionHistory (list): The history of observations selected from the candidate pool.
            - SimulationParameters (dict): The key input parameters used for this simulation run.
            - ElapsedTime (float): The total execution time in seconds for this iteration."""

    ### Set Up ###
    StartTime = time.time()
    random.seed(SimulationConfigInput["Seed"])
    np.random.seed(SimulationConfigInput["Seed"])

    ### Load Data ###
    DataFileInput = SimulationConfigInput["DataFileInput"]
    hl_trainer = None
    hl_cfg = None
    hl_data = None
    hl_model = None

    if DataFileInput == "hydralightning":
        hl_cfg = get_hl_cfg(SimulationConfigInput["add_useful_params"]["strategy_name"])
        hl_cfg["csv_path"] = os.path.join(
            f"{os.environ['PROJECT_ROOT']}/../henrihost-al/", hl_cfg["csv_path"]
        )
        print(f"< Instantiating hydralightning data and trainer >")
        datamodule = get_hl_datamodules(hl_cfg)

        hl_trainer, hl_model = get_hl_modules(hl_cfg, datamodule)

        print("===")

        # import Train data in pandas frame for active learning
        df_all_hl_dataset, y_size = full_datamodule_to_pd(datamodule)
        x_size = df_all_hl_dataset.shape[1] - y_size

        print("df_all_hl_dataset", df_all_hl_dataset.shape)

        # retrain df_full to the hl training set
        df_full = df_all_hl_dataset.loc[datamodule.train_data.labels, :]

        print("df_full", df_full.shape)
        # reserve df_test to the hl test set
        df_test = df_all_hl_dataset.loc[datamodule.test_data.labels, :]

        print("df_test", df_test.shape)
        ### Train Candidate Split ###
        df_Train, df_Candidate = TrainCandidateSplit_X(
            df_full.iloc[:, :x_size], SimulationConfigInput["CandidateProportion"]
        )

        df_Train = df_full.loc[df_Train.index, :]

        print("df_Train Split", df_Train.shape)
        print("df_Candidate Split", df_Candidate.shape)

        datamodule.train_data.update_indices(df_Train.index)

        hl_data = datamodule

        hl_trainer, hl_model, hl_data, hl_cfg = update_scheduler(
            hl_trainer, hl_model, hl_data, hl_cfg
        )

        print("===")

    else:
        y_size = 1
        # df_full = get_target_columns_except_first(DataFileInput)
        df_full = LoadData(DataFileInput)
        df_test = LoadData(DataFileInput)

        df_Train, df_Candidate = TrainCandidateSplit_X(
            df_full.iloc[:, y_size:], SimulationConfigInput["CandidateProportion"]
        )
        df_Train = df_full.iloc[df_Train.index, :]

    ### Update SimulationConfig Arguments ###

    SimulationConfigInput["hl_model"] = hl_model
    SimulationConfigInput["hl_data"] = hl_data
    SimulationConfigInput["hl_trainer"] = hl_trainer
    SimulationConfigInput["hl_cfg"] = hl_cfg
    SimulationConfigInput["y_size"] = y_size
    SimulationConfigInput["df_test"] = df_test
    SimulationConfigInput["df_full"] = df_full
    SimulationConfigInput["df_Train"] = df_Train
    SimulationConfigInput["df_Candidate"] = df_Candidate
    SimulationConfigInput["StartTime"] = StartTime

    ### Learning Process ###
    LearningProcedureOutput = LearningProcedure(SimulationConfigInputUpdated=SimulationConfigInput)

    ### Return Simulation Parameters ###
    SimulationParameters = {
        "DataFileInput": str(SimulationConfigInput["DataFileInput"]),
        "Seed": str(SimulationConfigInput["Seed"]),
        "CandidateProportion": str(SimulationConfigInput["CandidateProportion"]),
        "SelectorType": str(SimulationConfigInput["SelectorType"]),
        "ModelType": str(SimulationConfigInput["ModelType"]),
    }

    ### Return Time ###
    ElapsedTime = time.time() - StartTime

    ### Return Dictionary ###
    ErrorVecs = pd.DataFrame(LearningProcedureOutput["ErrorVecs"])

    SimulationResults = {
        "ErrorVecs": ErrorVecs,
        "SelectionHistory": LearningProcedureOutput["SelectedObservationHistory"],
        "WeightHistory": LearningProcedureOutput["WeightHistory"],
        "InitialTrainIndices": LearningProcedureOutput["InitialTrainIndices"],
        "SimulationParameters": SimulationParameters,
        "ElapsedTime": ElapsedTime,
        "k_top_candidate": SimulationConfigInput["add_useful_params"]["k_top_candidate"],
    }

    if "df_full" in SimulationConfigInput:
        SimulationResults["TotalPoolSize"] = len(SimulationConfigInput["df_full"])

    return SimulationResults
