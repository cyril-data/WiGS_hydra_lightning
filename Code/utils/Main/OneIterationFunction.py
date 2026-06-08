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
from utils.Prediction.LightHydra import get_hl_cfg, get_hl_modules

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

    if DataFileInput == "hydralightning":
        hl_cfg = get_hl_cfg()
        print(f"< Instantiating hydralightning data and trainer >")
        datamodule, hl_trainer = get_hl_modules(hl_cfg)

        # ========================================================
        # *** import data in pandas frame for active learning ***
        # train
        full_X = datamodule.train_data.data_x
        full_y_reg = datamodule.train_data.data_y_reg
        full_y_time_reg = datamodule.train_data.data_y_time_reg
        full_y_cls = datamodule.train_data.data_y_cls
        full_y_time_cls = datamodule.train_data.data_y_time_cls
        full_y = np.concatenate([full_y_reg, full_y_time_reg, full_y_cls, full_y_time_cls], axis=1)
        y_labels = (
            datamodule.train_data.y_reg_labels
            + datamodule.train_data.y_time_reg_labels
            + datamodule.train_data.y_cls_labels
            + datamodule.train_data.y_time_cls_labels
        )
        full_y_X = np.concatenate([full_y, full_X], axis=1)
        y_X_labels = datamodule.train_data.x_labels + y_labels
        df_full = pd.DataFrame(full_y_X, columns=y_X_labels)
        y_size = len(y_labels)
        # ***

        # *** import data in pandas frame for active learning ***
        # val
        full_X = datamodule.val_data.data_x
        full_y_reg = datamodule.val_data.data_y_reg
        full_y_time_reg = datamodule.val_data.data_y_time_reg
        full_y_cls = datamodule.val_data.data_y_cls
        full_y_time_cls = datamodule.val_data.data_y_time_cls
        full_y = np.concatenate([full_y_reg, full_y_time_reg, full_y_cls, full_y_time_cls], axis=1)
        y_labels = (
            datamodule.val_data.y_reg_labels
            + datamodule.val_data.y_time_reg_labels
            + datamodule.val_data.y_cls_labels
            + datamodule.val_data.y_time_cls_labels
        )
        full_y_X = np.concatenate([full_y, full_X], axis=1)
        y_X_labels = datamodule.val_data.x_labels + y_labels
        df_test = pd.DataFrame(full_y_X, columns=y_X_labels)
        y_size = len(y_labels)
        # ***
        # ========================================================

    else:
        y_size = 1
        # df_full = get_target_columns_except_first(DataFileInput)
        df_full = LoadData(DataFileInput)
        df_test = LoadData(DataFileInput)

    ### Train Candidate Split ###
    df_Train, df_Candidate = TrainCandidateSplit_X(
        df_full.iloc[:, y_size:], SimulationConfigInput["CandidateProportion"]
    )

    df_Train = df_full.iloc[df_Train.index, :]

    ### Update SimulationConfig Arguments ###

    SimulationConfigInput["hl_data"] = hl_data
    SimulationConfigInput["hl_trainer"] = hl_trainer
    SimulationConfigInput["hl_cfg"] = hl_cfg
    SimulationConfigInput["y_size"] = y_size
    SimulationConfigInput["df_test"] = df_test
    SimulationConfigInput["df_full"] = df_full
    SimulationConfigInput["df_Train"] = df_Train
    SimulationConfigInput["df_Candidate"] = df_Candidate

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
    }
    return SimulationResults
