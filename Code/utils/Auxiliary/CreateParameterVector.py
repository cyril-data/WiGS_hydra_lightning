### Import packages ###
import itertools
import pandas as pd
import numpy as np

# Data: Iris  MONK1  MONK3  Bar7 (10)  COMPAS (50) | BankNote (10)  BreastCancer (5)  CarEvaluation (10)  FICO (50)  Haberman
def CreateParameterVectorFunction(Data,
                                  Seed,                     # range(0,50)
                                  RashomonThreshold,        # For TreeFarms
                                  DiversityWeight,          # For BatchQBC
                                  DensityWeight,            # For BatchQBC
                                  BatchSize,                # For all batch selectors
                                  Partition,                # SLURM partition
                                  Time,                     # SLURM time limit
                                  Memory,                   # SLURM memory limit
                                  IncludePL_RF=False,       # Passive Learning with RandomForestClassifierPredictor
                                  IncludePL_GPC=False,      # Passive Learning with GaussianProcessClassifierPredictor
                                  IncludePL_BNN=False,      # Passive Learning with BayesianNeuralNetworkPredictor
                                  IncludeBALD_BNN=False,    # BALD with BayesianNeuralNetworkPredictor
                                  IncludeBALD_GPC=False,    # BALD with GaussianProcessClassifierPredictor
                                  IncludeQBC_TreeFarms_Unique=False, # BatchQBC with TreeFarmsPredictor (UniqueErrorsInput=1) -> UNREAL
                                  IncludeQBC_TreeFarms_Duplicate=False, # BatchQBC with TreeFarmsPredictor (UniqueErrorsInput=0) -> DUREAL
                                  IncludeQBC_RF=False,      # BatchQBC with RandomForestClassifierPredictor
                                  IncludeLFR_TreeFarms_Unique=False, # NEW: BatchQBC with LFRPredictor (UniqueErrorsInput=1) -> UNREAL_LFR
                                  IncludeLFR_TreeFarms_Duplicate=False, # NEW: BatchQBC with LFRPredictor (UniqueErrorsInput=0) -> DUREAL_LFR
                                  auto_tune_epsilon_for_lfr=True):

    ### Data Abbreviations ###
    AbbreviationDictionary = {"BankNote": "BN",
                              "Bar7": "B7",
                              "BreastCancer": "BC",
                              "CarEvaluation": "CE",
                              "COMPAS": "CP",
                              "FICO": "FI",
                              "Haberman": "HM",
                              "Iris": "IS",
                              "MONK1": "M1",
                              "MONK3":"M3"}
    JobNameAbbrev = AbbreviationDictionary[Data]

    # Parameter Dictionary #
    all_parameter_dicts = []

    ### Base Parameter Dictionary (unused if specific flags are set) ###
    base_params = {
        "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
        "SelectorType": ["BatchQBCSelector"], "ModelType": ["TreeFarmsPredictor"], "UniqueErrorsInput": [0],
        "n_estimators": [100], "regularization": [0.01], "RashomonThresholdType": ["Adder"],
        "RashomonThreshold": [RashomonThreshold], "Type": ["Classification"], "DiversityWeight": [DiversityWeight],
        "DensityWeight": [DensityWeight], "BatchSize": [BatchSize], "Partition": [Partition],
        "Time": [Time], "Memory": [Memory]
    }

    ### 1. PassiveLearningSelector and RandomForestClassifierPredictor (RF_PL) ###
    if IncludePL_RF:
        PL_RF_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"], "ModelType": ["RandomForestClassifierPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [0], "DensityWeight": [0], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(PL_RF_ParameterDictionary)

    ### 2. PassiveLearningSelector and GaussianProcessClassifierPredictor (GPC_PL) ###
    if IncludePL_GPC:
        PL_GPC_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"], "ModelType": ["GaussianProcessClassifierPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [0], "regularization": [0.0],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [0], "DensityWeight": [0], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "kernel_type": ['RBF'], "kernel_length_scale": [1.0], "kernel_nu": [1.5],
            "optimizer": ['fmin_l_bfgs_b'], "n_restarts_optimizer": [0], "max_iter_predict": [100],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(PL_GPC_ParameterDictionary)

    ### 3. PassiveLearningSelector and BayesianNeuralNetworkPredictor (BNN_PL) ###
    if IncludePL_BNN:
        PL_BNN_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["PassiveLearningSelector"], "ModelType": ["BayesianNeuralNetworkPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [0], "regularization": [0.0],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [0], "DensityWeight": [0], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "hidden_size": [50], "dropout_rate": [0.2], "epochs": [100],
            "learning_rate": [0.001], "batch_size_train": [32], "K_BALD_Samples": [20],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(PL_BNN_ParameterDictionary)

    ### 4. BALDSelector and BayesianNeuralNetworkPredictor (BNN_BALD) ###
    if IncludeBALD_BNN:
        BALD_BNN_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"], "ModelType": ["BayesianNeuralNetworkPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [0], "regularization": [0.0],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [0], "DensityWeight": [0], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "hidden_size": [50], "dropout_rate": [0.2], "epochs": [100],
            "learning_rate": [0.001], "batch_size_train": [32], "K_BALD_Samples": [20],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(BALD_BNN_ParameterDictionary)

    ### 5. BALDSelector and GaussianProcessClassifierPredictor (GPC_BALD) ###
    if IncludeBALD_GPC:
        BALD_GPC_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BALDSelector"], "ModelType": ["GaussianProcessClassifierPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [0], "regularization": [0.0],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [0], "DensityWeight": [0], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "kernel_type": ['RBF'], "kernel_length_scale": [1.0], "kernel_nu": [1.5],
            "optimizer": ['fmin_l_bfgs_b'], "n_restarts_optimizer": [0], "max_iter_predict": [100],
            "K_BALD_Samples": [20],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(BALD_GPC_ParameterDictionary)

    ### 6. UNREAL: BatchQBCSelector and TreeFarmsPredictor (UniqueErrorsInput=1) ###
    if IncludeQBC_TreeFarms_Unique:
        UNREAL_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], "ModelType": ["TreeFarmsPredictor"],
            "UniqueErrorsInput": [1], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [RashomonThreshold],
            "Type": ["Classification"], "DiversityWeight": [DiversityWeight], "DensityWeight": [DensityWeight],
            "BatchSize": [BatchSize], "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(UNREAL_ParameterDictionary)

    ### 7. DUREAL: BatchQBCSelector and TreeFarmsPredictor (UniqueErrorsInput=0) ###
    if IncludeQBC_TreeFarms_Duplicate:
        DUREAL_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], "ModelType": ["TreeFarmsPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [RashomonThreshold],
            "Type": ["Classification"], "DiversityWeight": [DiversityWeight], "DensityWeight": [DensityWeight],
            "BatchSize": [BatchSize], "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(DUREAL_ParameterDictionary)

    ### 8. BatchQBCSelector with RandomForestClassifierPredictor (RF_QBC) ###
    if IncludeQBC_RF:
        QBC_RF_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], "ModelType": ["RandomForestClassifierPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [0], "Type": ["Classification"],
            "DiversityWeight": [DiversityWeight], "DensityWeight": [DensityWeight], "BatchSize": [BatchSize],
            "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [False]
        }
        all_parameter_dicts.append(QBC_RF_ParameterDictionary)

    ### 9. UNREAL_LFR: BatchQBCSelector with LFRPredictor (UniqueErrorsInput=1) ###
    if IncludeLFR_TreeFarms_Unique:
        UNREAL_LFR_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], "ModelType": ["LFRPredictor"],
            "UniqueErrorsInput": [1], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [RashomonThreshold],
            "Type": ["Classification"], "DiversityWeight": [DiversityWeight], "DensityWeight": [DensityWeight],
            "BatchSize": [BatchSize], "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [auto_tune_epsilon_for_lfr]
        }
        all_parameter_dicts.append(UNREAL_LFR_ParameterDictionary)

    ### 10. DUREAL_LFR: BatchQBCSelector with LFRPredictor (UniqueErrorsInput=0) ###
    if IncludeLFR_TreeFarms_Duplicate:
        DUREAL_LFR_ParameterDictionary = {
            "Data": [Data], "Seed": list(Seed), "TestProportion": [0.2], "CandidateProportion": [0.8],
            "SelectorType": ["BatchQBCSelector"], "ModelType": ["LFRPredictor"],
            "UniqueErrorsInput": [0], "n_estimators": [100], "regularization": [0.01],
            "RashomonThresholdType": ["Adder"], "RashomonThreshold": [RashomonThreshold],
            "Type": ["Classification"], "DiversityWeight": [DiversityWeight], "DensityWeight": [DensityWeight],
            "BatchSize": [BatchSize], "Partition": [Partition], "Time": [Time], "Memory": [Memory],
            "auto_tune_epsilon": [auto_tune_epsilon_for_lfr]
        }
        all_parameter_dicts.append(DUREAL_LFR_ParameterDictionary)

    # Combine all parameter dictionaries into a single DataFrame
    if not all_parameter_dicts:
        return pd.DataFrame()

    list_of_dfs = []
    for p_dict in all_parameter_dicts:
        list_of_dfs.append(pd.DataFrame.from_records(itertools.product(*p_dict.values()), columns=p_dict.keys()))

    ParameterVector = pd.concat(list_of_dfs, ignore_index=True)

    # Ensure all possible columns are present, filling NaNs for missing model-specific params
    all_possible_columns = sorted(list(set(col for d in all_parameter_dicts for col in d.keys())))
    ParameterVector = ParameterVector.reindex(columns=all_possible_columns)

    numeric_cols = ParameterVector.select_dtypes(include=np.number).columns
    ParameterVector[numeric_cols] = ParameterVector[numeric_cols].fillna(0)
    object_cols = ParameterVector.select_dtypes(include='object').columns
    ParameterVector[object_cols] = ParameterVector[object_cols].fillna('')

    # Sort and re-index
    ParameterVector = ParameterVector.sort_values("Seed")
    ParameterVector.index = range(0, ParameterVector.shape[0])

    #############################################################################
    ### Job and Output Name (REFACTORED) ###
    #############################################################################

    # 1. Define the new, shorter method names based on model/selector combinations
    conditions = [
        (ParameterVector["ModelType"] == "RandomForestClassifierPredictor") & (ParameterVector["SelectorType"] == "PassiveLearningSelector"),
        (ParameterVector["ModelType"] == "GaussianProcessClassifierPredictor") & (ParameterVector["SelectorType"] == "PassiveLearningSelector"),
        (ParameterVector["ModelType"] == "BayesianNeuralNetworkPredictor") & (ParameterVector["SelectorType"] == "PassiveLearningSelector"),
        (ParameterVector["ModelType"] == "BayesianNeuralNetworkPredictor") & (ParameterVector["SelectorType"] == "BALDSelector"),
        (ParameterVector["ModelType"] == "GaussianProcessClassifierPredictor") & (ParameterVector["SelectorType"] == "BALDSelector"),
        (ParameterVector["ModelType"] == "TreeFarmsPredictor") & (ParameterVector["SelectorType"] == "BatchQBCSelector") & (ParameterVector["UniqueErrorsInput"] == 1),
        (ParameterVector["ModelType"] == "TreeFarmsPredictor") & (ParameterVector["SelectorType"] == "BatchQBCSelector") & (ParameterVector["UniqueErrorsInput"] == 0),
        (ParameterVector["ModelType"] == "RandomForestClassifierPredictor") & (ParameterVector["SelectorType"] == "BatchQBCSelector"),
        (ParameterVector["ModelType"] == "LFRPredictor") & (ParameterVector["SelectorType"] == "BatchQBCSelector") & (ParameterVector["UniqueErrorsInput"] == 1),
        (ParameterVector["ModelType"] == "LFRPredictor") & (ParameterVector["SelectorType"] == "BatchQBCSelector") & (ParameterVector["UniqueErrorsInput"] == 0),
    ]
    choices = [
        "RF_PL", "GPC_PL", "BNN_PL", "BNN_BALD", "GPC_BALD",
        "UNREAL_UEI1", "DUREAL_UEI0", "RF_QBC_UEI0",
        "Ulfr_UEI1", "Dlfr_UEI0"
    ]
    ParameterVector["MethodName"] = np.select(conditions, choices, default="UNKNOWN")
    
    # 1b. ***NEW*** Inject the AT parameter into the MethodName for LFR models
    lfr_model_mask = (ParameterVector["ModelType"] == "LFRPredictor")
    if 'auto_tune_epsilon' in ParameterVector.columns and lfr_model_mask.any():
        # Create the AT string (e.g., "_AT0" or "_AT1") for LFR rows
        at_string = "_AT" + ParameterVector.loc[lfr_model_mask, "auto_tune_epsilon"].astype(int).astype(str)
        # Split MethodName (e.g., "Ulfr_UEI1") and insert the AT string
        method_parts = ParameterVector.loc[lfr_model_mask, "MethodName"].str.split('_', n=1, expand=True)
        # Recombine to create the new name (e.g., "Ulfr_AT1_UEI1")
        ParameterVector.loc[lfr_model_mask, "MethodName"] = method_parts[0] + at_string + "_" + method_parts[1]

    # 2. Build the JobName starting with Seed, Data, and the new MethodName
    ParameterVector["JobName"] = (
        ParameterVector["Seed"].astype(str) +
        ParameterVector["Data"].map(AbbreviationDictionary).astype(str) + "_" +
        ParameterVector["MethodName"]
    )

    # 3. Conditionally append relevant parameters for specific methods
    # QBC-specific parameters (for UNREAL, DUREAL, RF_QBC, LFR variants)
    qbc_mask = ParameterVector["SelectorType"] == "BatchQBCSelector"
    # RashomonThreshold is only used by TreeFarms/LFR
    rashomon_mask = qbc_mask & ParameterVector["ModelType"].isin(["TreeFarmsPredictor", "LFRPredictor"])
    ParameterVector.loc[rashomon_mask, "JobName"] += "_A" + ParameterVector.loc[rashomon_mask, "RashomonThreshold"].astype(str)
    # Diversity and Density weights are used by all QBC methods
    ParameterVector.loc[qbc_mask, "JobName"] += "_DW" + ParameterVector.loc[qbc_mask, "DiversityWeight"].astype(str)
    ParameterVector.loc[qbc_mask, "JobName"] += "_DEW" + ParameterVector.loc[qbc_mask, "DensityWeight"].astype(str)

    # All methods have a BatchSize
    ParameterVector["JobName"] += "_B" + ParameterVector["BatchSize"].astype(str)

    # ***REMOVED*** The old block for appending _AT at the end is no longer here.

    # BNN-specific hyperparameters
    bnn_mask = ParameterVector["ModelType"] == "BayesianNeuralNetworkPredictor"
    if 'hidden_size' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_HS" + ParameterVector.loc[bnn_mask, "hidden_size"].astype(str)
    if 'dropout_rate' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_DR" + ParameterVector.loc[bnn_mask, "dropout_rate"].astype(str)
    if 'epochs' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_E" + ParameterVector.loc[bnn_mask, "epochs"].astype(str)
    if 'learning_rate' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_LR" + ParameterVector.loc[bnn_mask, "learning_rate"].astype(str)
    if 'batch_size_train' in ParameterVector.columns:
        ParameterVector.loc[bnn_mask, "JobName"] += "_BST" + ParameterVector.loc[bnn_mask, "batch_size_train"].astype(str)

    # GPC-specific hyperparameters
    gpc_mask = ParameterVector["ModelType"] == "GaussianProcessClassifierPredictor"
    if 'kernel_type' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KT" + ParameterVector.loc[gpc_mask, "kernel_type"].astype(str)
    if 'kernel_length_scale' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KLS" + ParameterVector.loc[gpc_mask, "kernel_length_scale"].astype(str)
    if 'kernel_nu' in ParameterVector.columns:
        ParameterVector.loc[gpc_mask, "JobName"] += "_KNU" + ParameterVector.loc[gpc_mask, "kernel_nu"].astype(str)

    # BALD-specific `K_BALD_Samples`
    bald_mask = ParameterVector["SelectorType"] == "BALDSelector"
    if 'K_BALD_Samples' in ParameterVector.columns:
        # Only add K if it's a non-zero value provided for BALD runs
        k_mask = bald_mask & (ParameterVector['K_BALD_Samples'] > 0)
        ParameterVector.loc[k_mask, "JobName"] += "_K" + ParameterVector.loc[k_mask, "K_BALD_Samples"].astype(str)

    # 4. Final cleaning of the JobName string
    ParameterVector["JobName"] = (
        ParameterVector["JobName"]
        .str.replace(r"0\.(?=\d)", "", regex=True)  # "0.5" -> "5"
        .str.replace(r"\.0(?!\d)", "", regex=True)  # "10.0" -> "10"
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )

    # 5. Generate Output Name based on the full ModelType (predictor name) for file path compatibility
    ParameterVector["Output"] = (
        ParameterVector["Data"].astype(str) + "/" +
        ParameterVector["ModelType"].astype(str) + "/Raw/" + # Group by full predictor name for file path
        ParameterVector["JobName"] + ".pkl"
    )

    ### Find Missing Simulations (Optional) ###
    ### Return ###
    return ParameterVector

### FilterJobNames ###
def FilterJobNames(df, filter_strings):
    mask = df['JobName'].apply(lambda x: any(filter_str in x for filter_str in filter_strings))
    return df[mask]