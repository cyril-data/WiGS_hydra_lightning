### Packages ###
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import os
import warnings

from utils.Auxiliary.LoadAnalyzedData import LoadAnalyzedData
from utils.Auxiliary.MeanVariancePlot import MeanVariancePlot

### Analyze Results Function ###
def AnalyzeResultsFunction(DataType, methods_to_plot=None):
    """
    Analyzes and plots results for various active learning simulation methods.
    ...
    """

    ### Load Data ###
    BaseDirectory = os.path.join(os.path.expanduser("~"), "Documents", "RashomonActiveLearning", "Results")

    method_configs = {
        "RF_PL":          {"ModelDir": "RandomForestClassifierPredictor", "FilePrefix": "_RF_PL"},
        "RF_QBC":         {"ModelDir": "RandomForestClassifierPredictor", "FilePrefix": "_RF_QBC"},
        "GPC_PL":         {"ModelDir": "GaussianProcessClassifierPredictor", "FilePrefix": "_GPC_PL"},
        "GPC_BALD":       {"ModelDir": "GaussianProcessClassifierPredictor", "FilePrefix": "_GPC_BALD"},
        "BNN_PL":         {"ModelDir": "BayesianNeuralNetworkPredictor", "FilePrefix": "_BNN_PL"},
        "BNN_BALD":       {"ModelDir": "BayesianNeuralNetworkPredictor", "FilePrefix": "_BNN_BALD"},
        "UNREAL":         {"ModelDir": "TreeFarmsPredictor", "FilePrefix": "_UNREAL"},
        "DUREAL" :        {"ModelDir": "TreeFarmsPredictor", "FilePrefix": "_DUREAL"},
        "UNREAL_LFR_AT1": {"ModelDir": "LFRPredictor", "FilePrefix": "_Ulfr_AT1"},
        "DUREAL_LFR_AT1": {"ModelDir": "LFRPredictor", "FilePrefix": "_Dlfr_AT1"},
        "UNREAL_LFR_AT0": {"ModelDir": "LFRPredictor", "FilePrefix": "_Ulfr_AT0"},
        "DUREAL_LFR_AT0": {"ModelDir": "LFRPredictor", "FilePrefix": "_Dlfr_AT0"},
    }

    loaded_data_by_method = {}
    raw_data_tables = {}
    for method_key, config in method_configs.items():
        if not config["FilePrefix"]:
            warnings.warn(f"No FilePrefix provided for method '{method_key}'. Skipping.")
            continue
        data = LoadAnalyzedData(data_type=DataType, base_directory=BaseDirectory, model_directory=config["ModelDir"], file_prefix=config["FilePrefix"])
        loaded_data_by_method[method_key] = data
        raw_data_tables[method_key] = data

        # --- THIS IS THE CORRECTED LINE ---
        if all(value is None for value in data.values()):
             warnings.warn(f"No data files were loaded for method '{method_key}' using prefix '{config['FilePrefix']}'.")


    ### Shape Table ###
    ShapeTable = {key: data["Error"].shape[0] for key, data in loaded_data_by_method.items() if data.get("Error") is not None}
    ShapeTable = pd.DataFrame([ShapeTable]) if ShapeTable else pd.DataFrame()

    ### Time Table ###
    time_data_list = []
    for key in method_configs.keys():
        data = loaded_data_by_method.get(key, {})
        row_data = {"Method": key}
        
        if data.get("Time") is not None and not data["Time"].empty:
            time_in_seconds = data["Time"].iloc[:, 0]
            row_data["Mean (minutes)"] = float(f"{time_in_seconds.mean() / 60:.2f}")
            row_data["Max (minutes)"] = float(f"{time_in_seconds.max() / 60:.2f}")
        else:
            row_data["Mean (minutes)"] = np.nan
            row_data["Max (minutes)"] = np.nan
        time_data_list.append(row_data)

    if time_data_list:
        TimeTable = pd.DataFrame(time_data_list).set_index("Method")
    else:
        TimeTable = pd.DataFrame()

    ### Legend and Styling Definitions ###
    PlotSubtitle = f"Dataset: {DataType}"
    colors = {
        "RF_PL": "black", "GPC_PL": "gray", "BNN_PL": "silver",
        "BNN_BALD": "darkviolet", "GPC_BALD": "mediumorchid", "RF_QBC": "green",
        "UNREAL": "firebrick", "DUREAL": "gold",
        "UNREAL_LFR_AT1": "dodgerblue", "DUREAL_LFR_AT1": "darkorange",
        "UNREAL_LFR_AT0": "cyan", "DUREAL_LFR_AT0": "sandybrown"
    }
    linestyles = {method: "solid" for method in method_configs.keys()}
    LegendMapping = {
        "RF_PL": "Passive Learning (RF)", "GPC_PL": "Passive Learning (GPC)", "BNN_PL": "Passive Learning (BNN)",
        "BNN_BALD": "BALD (BNN)", "GPC_BALD": "BALD (GPC)", "RF_QBC": "QBC (RF)",
        "UNREAL": "UNREAL", "DUREAL": "DUREAL",
        "UNREAL_LFR_AT1": "UNREAL_LFR (AT=1)", "DUREAL_LFR_AT1": "DUREAL_LFR (AT=1)",
        "UNREAL_LFR_AT0": "UNREAL_LFR (AT=0)", "DUREAL_LFR_AT0": "DUREAL_LFR (AT=0)"
    }
    
    ### Trace Plot (F1 Score) ###
    if methods_to_plot is None:
        methods_to_include = [key for key, data in loaded_data_by_method.items() if data.get("Error") is not None]
    else:
        methods_to_include = []
        for method in methods_to_plot:
            if method in loaded_data_by_method and loaded_data_by_method[method].get("Error") is not None:
                methods_to_include.append(method)
            else:
                warnings.warn(f"Requested method '{method}' for F1 plot not found or has no 'Error' data. It will be skipped.")
    
    error_data_for_plot = {key: loaded_data_by_method[key]["Error"] for key in methods_to_include}

    TracePlotMean, TracePlotVariance = None, None
    if not error_data_for_plot:
        warnings.warn("No valid data to plot for F1 score after filtering.")
    else:
        TracePlotMean, TracePlotVariance = MeanVariancePlot(RelativeError=None, Colors=colors, LegendMapping=LegendMapping, Linestyles=linestyles, Y_Label="F1 Score", Subtitle=PlotSubtitle, TransparencyVal=0.05, VarInput=True, CriticalValue=1.96, **error_data_for_plot)
        if TracePlotMean: TracePlotMean.get_axes()[0].legend(loc="best"); plt.close(TracePlotMean)
        if TracePlotVariance: TracePlotVariance.get_axes()[0].legend(loc="best"); plt.close(TracePlotVariance)

    ### Refit Frequency Plot ###
    refit_plot_data = {}
    refit_methods_to_plot = [
        "UNREAL_LFR_AT1", 
        # "DUREAL_LFR_AT1", 
        # "UNREAL", 
        # "DUREAL", 
        "UNREAL_LFR_AT0", 
        # "DUREAL_LFR_AT0"
        ]

    for key in refit_methods_to_plot:
        if key in loaded_data_by_method and loaded_data_by_method[key].get("RefitDecision") is not None:
            refit_plot_data[key] = loaded_data_by_method[key]["RefitDecision"]
        else:
            warnings.warn(f"RefitDecision data for '{key}' not found. It will be skipped in the RefitFrequencyPlot.")

    RefitFrequencyPlot = None
    if not refit_plot_data:
        warnings.warn("No RefitDecision data found for UNREAL/DUREAL variants.")
    else:
        RefitFrequencyPlot = MeanVariancePlot(
            RelativeError=None, Colors=colors, LegendMapping=LegendMapping, Linestyles=linestyles, 
            Y_Label="Refit Frequency", Subtitle=f"Dataset: {DataType} - Refit Behavior",
            TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **refit_plot_data
        )
        if RefitFrequencyPlot: RefitFrequencyPlot.get_axes()[0].legend(loc="best"); plt.close(RefitFrequencyPlot)

    ### Number of Trees Plots ###
    # This section remains correct.
    
    # --- Plot 1: UNREAL_LFR_AT1 ---
    plot_data, plot_legend, plot_colors = {}, {}, {}
    TreePlot_UNREAL_LFR_AT1 = None
    data = loaded_data_by_method.get("UNREAL_LFR_AT1")
    if data:
        if data.get("AllTreeCount") is not None: plot_data["Total"] = np.log(data["AllTreeCount"].replace(0, 1)); plot_legend["Total"] = "Total Trees"; plot_colors["Total"] = "darkorange"
        if data.get("UniqueTreeCount") is not None: plot_data["Unique"] = np.log(data["UniqueTreeCount"].replace(0, 1)); plot_legend["Unique"] = "Unique Trees"; plot_colors["Unique"] = "dodgerblue"
        if plot_data:
            TreePlot_UNREAL_LFR_AT1 = MeanVariancePlot(RelativeError=None, Colors=plot_colors, LegendMapping=plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - UNREAL_LFR (AT=1)", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **plot_data)
            if TreePlot_UNREAL_LFR_AT1: TreePlot_UNREAL_LFR_AT1.get_axes()[0].legend(loc="best"); plt.close(TreePlot_UNREAL_LFR_AT1)

    # --- Plot 2: DUREAL_LFR_AT1 ---
    plot_data, plot_legend, plot_colors = {}, {}, {}
    TreePlot_DUREAL_LFR_AT1 = None
    data = loaded_data_by_method.get("DUREAL_LFR_AT1")
    if data:
        if data.get("AllTreeCount") is not None: plot_data["Total"] = np.log(data["AllTreeCount"].replace(0, 1)); plot_legend["Total"] = "Total Trees"; plot_colors["Total"] = "darkorange"
        if data.get("UniqueTreeCount") is not None: plot_data["Unique"] = np.log(data["UniqueTreeCount"].replace(0, 1)); plot_legend["Unique"] = "Unique Trees"; plot_colors["Unique"] = "dodgerblue"
        if plot_data:
            TreePlot_DUREAL_LFR_AT1 = MeanVariancePlot(RelativeError=None, Colors=plot_colors, LegendMapping=plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - DUREAL_LFR (AT=1)", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **plot_data)
            if TreePlot_DUREAL_LFR_AT1: TreePlot_DUREAL_LFR_AT1.get_axes()[0].legend(loc="best"); plt.close(TreePlot_DUREAL_LFR_AT1)

    # --- Plot 3: UNREAL_LFR_AT0 ---
    plot_data, plot_legend, plot_colors = {}, {}, {}
    TreePlot_UNREAL_LFR_AT0 = None
    data = loaded_data_by_method.get("UNREAL_LFR_AT0")
    if data:
        if data.get("AllTreeCount") is not None: plot_data["Total"] = np.log(data["AllTreeCount"].replace(0, 1)); plot_legend["Total"] = "Total Trees"; plot_colors["Total"] = "darkorange"
        if data.get("UniqueTreeCount") is not None: plot_data["Unique"] = np.log(data["UniqueTreeCount"].replace(0, 1)); plot_legend["Unique"] = "Unique Trees"; plot_colors["Unique"] = "dodgerblue"
        if plot_data:
            TreePlot_UNREAL_LFR_AT0 = MeanVariancePlot(RelativeError=None, Colors=plot_colors, LegendMapping=plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - UNREAL_LFR (AT=0)", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **plot_data)
            if TreePlot_UNREAL_LFR_AT0: TreePlot_UNREAL_LFR_AT0.get_axes()[0].legend(loc="best"); plt.close(TreePlot_UNREAL_LFR_AT0)

    # --- Plot 4: DUREAL_LFR_AT0 ---
    plot_data, plot_legend, plot_colors = {}, {}, {}
    TreePlot_DUREAL_LFR_AT0 = None
    data = loaded_data_by_method.get("DUREAL_LFR_AT0")
    if data:
        if data.get("AllTreeCount") is not None: plot_data["Total"] = np.log(data["AllTreeCount"].replace(0, 1)); plot_legend["Total"] = "Total Trees"; plot_colors["Total"] = "darkorange"
        if data.get("UniqueTreeCount") is not None: plot_data["Unique"] = np.log(data["UniqueTreeCount"].replace(0, 1)); plot_legend["Unique"] = "Unique Trees"; plot_colors["Unique"] = "dodgerblue"
        if plot_data:
            TreePlot_DUREAL_LFR_AT0 = MeanVariancePlot(RelativeError=None, Colors=plot_colors, LegendMapping=plot_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - DUREAL_LFR (AT=0)", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **plot_data)
            if TreePlot_DUREAL_LFR_AT0: TreePlot_DUREAL_LFR_AT0.get_axes()[0].legend(loc="best"); plt.close(TreePlot_DUREAL_LFR_AT0)


    # --- Plot 5: UNREAL (from TreeFarms) ---
    unreal_tf_plot_data, unreal_tf_legend, unreal_tf_colors = {}, {}, {}
    TreePlot_UNREAL_TF = None
    unreal_tf_data = loaded_data_by_method.get("UNREAL")
    if unreal_tf_data:
        if unreal_tf_data.get("AllTreeCount") is not None: unreal_tf_plot_data["Total"] = np.log(unreal_tf_data["AllTreeCount"].replace(0, 1)); unreal_tf_legend["Total"] = "Total Trees"; unreal_tf_colors["Total"] = "darkorange"
        if unreal_tf_data.get("UniqueTreeCount") is not None: unreal_tf_plot_data["Unique"] = np.log(unreal_tf_data["UniqueTreeCount"].replace(0, 1)); unreal_tf_legend["Unique"] = "Unique Trees"; unreal_tf_colors["Unique"] = "dodgerblue"
        if unreal_tf_plot_data:
            TreePlot_UNREAL_TF = MeanVariancePlot(RelativeError=None, Colors=unreal_tf_colors, LegendMapping=unreal_tf_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - UNREAL", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **unreal_tf_plot_data)
            if TreePlot_UNREAL_TF: TreePlot_UNREAL_TF.get_axes()[0].legend(loc="best"); plt.close(TreePlot_UNREAL_TF)

    # --- Plot 6: DUREAL (from TreeFarms) ---
    dureal_tf_plot_data, dureal_tf_legend, dureal_tf_colors = {}, {}, {}
    TreePlot_DUREAL_TF = None
    dureal_tf_data = loaded_data_by_method.get("DUREAL")
    if dureal_tf_data:
        if dureal_tf_data.get("AllTreeCount") is not None: dureal_tf_plot_data["Total"] = np.log(dureal_tf_data["AllTreeCount"].replace(0, 1)); dureal_tf_legend["Total"] = "Total Trees"; dureal_tf_colors["Total"] = "darkorange"
        if dureal_tf_data.get("UniqueTreeCount") is not None: dureal_tf_plot_data["Unique"] = np.log(dureal_tf_data["UniqueTreeCount"].replace(0, 1)); dureal_tf_legend["Unique"] = "Unique Trees"; dureal_tf_colors["Unique"] = "dodgerblue"
        if dureal_tf_data:
            TreePlot_DUREAL_TF = MeanVariancePlot(RelativeError=None, Colors=dureal_tf_colors, LegendMapping=dureal_tf_legend, Linestyles={'Total':'solid', 'Unique':'solid'}, Y_Label="log(Number of Trees)", Subtitle=f"Dataset: {DataType} - DUREAL", TransparencyVal=0.05, VarInput=False, CriticalValue=1.96, **dureal_tf_plot_data)
            if TreePlot_DUREAL_TF: TreePlot_DUREAL_TF.get_axes()[0].legend(loc="best"); plt.close(TreePlot_DUREAL_TF)


    ### Output ###
    return {
        "TracePlotMean": TracePlotMean,
        "TracePlotVariance": TracePlotVariance,
        "RefitFrequencyPlot": RefitFrequencyPlot,
        "TreePlot_UNREAL_LFR_AT1": TreePlot_UNREAL_LFR_AT1,
        "TreePlot_DUREAL_LFR_AT1": TreePlot_DUREAL_LFR_AT1,
        "TreePlot_UNREAL_LFR_AT0": TreePlot_UNREAL_LFR_AT0,
        "TreePlot_DUREAL_LFR_AT0": TreePlot_DUREAL_LFR_AT0,
        "TreePlot_UNREAL_TF": TreePlot_UNREAL_TF,
        "TreePlot_DUREAL_TF": TreePlot_DUREAL_TF,
        "ShapeTable": ShapeTable,
        "TimeTable": TimeTable,
        "RawData": raw_data_tables
    }