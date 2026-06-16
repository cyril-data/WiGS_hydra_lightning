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
    train_datamodule_to_pd,
    val_datamodule_to_pd,
    update_scheduler,
)

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
        hl_cfg = get_hl_cfg()
        print(f"< Instantiating hydralightning data and trainer >")
        datamodule = get_hl_datamodules(hl_cfg)
        hl_trainer, hl_model = get_hl_modules(hl_cfg, datamodule)

        # import Train data in pandas frame for active learning
        df_full, y_size = train_datamodule_to_pd(datamodule)
        x_size = df_full.shape[1] - y_size

        # import Val data in pandas frame for active learning
        df_test, _ = val_datamodule_to_pd(datamodule)

        ### Train Candidate Split ###
        df_Train, df_Candidate = TrainCandidateSplit_X(
            df_full.iloc[:, :x_size], SimulationConfigInput["CandidateProportion"]
        )

        df_Train = df_full.loc[df_Train.index, :]

        datamodule.train_data.update_indices(df_Train.index)

        hl_data = datamodule

        hl_trainer, hl_model, hl_data, hl_cfg = update_scheduler(
            hl_trainer, hl_model, hl_data, hl_cfg
        )

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
