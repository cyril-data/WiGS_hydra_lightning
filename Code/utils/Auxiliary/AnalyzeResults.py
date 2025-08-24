# ### Import Packages ###
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# ### Append Path ###
# import sys
# sys.path.append('..')

# ### Import functions ###
# from utils.Auxiliary import *

# def ProcessAllResults(results_dir, image_dir):
#     print(f"--- Starting Analysis ---")
#     print(f"Loading results from: {results_dir}")

#     ## Define aesthetics for all strategies ##
#     master_colors = {
#         'Passive Learning': 'gray', 
#         'GSx': 'cornflowerblue', 
#         'GSy': 'salmon', 'iGS': 'red',
#         'WiGS (Static w_x=0.75)': 'lightgreen', 
#         'WiGS (Static w_x=0.5)': 'forestgreen',
#         'WiGS (Static w_x=0.25)': 'darkgreen', 
#         'WiGS (Time-Decay, Linear)': 'orange',
#         'WiGS (Time-Decay, Exponential)': 'saddlebrown', 
#         'WiGS (MAB-UCB1, c=0.5)': 'orchid',
#         'WiGS (MAB-UCB1, c=2.0)': 'darkviolet', 
#         'WiGS (MAB-UCB1, c=5.0)': 'indigo'
#     }
#     master_linestyles = {
#         'Passive Learning': ':', 
#         'GSx': ':', 
#         'GSy': ':', 'iGS': '-',
#         'WiGS (Static w_x=0.75)': '-', 
#         'WiGS (Static w_x=0.5)': '-.',
#         'WiGS (Static w_x=0.25)': '--', 
#         'WiGS (Time-Decay, Linear)': '-',
#         'WiGS (Time-Decay, Exponential)': '-.', 
#         'WiGS (MAB-UCB1, c=0.5)': '-',
#         'WiGS (MAB-UCB1, c=2.0)': '-', 
#         'WiGS (MAB-UCB1, c=5.0)': '-'
#     }
#     master_legend = {
#         'Passive Learning': 'Random', 
#         'GSx': 'GSx', 
#         'GSy': 'GSy', 
#         'iGS': 'iGS',
#         'WiGS (Static w_x=0.75)': 'WiGS (Static, w_x=0.75)', 
#         'WiGS (Static w_x=0.5)': 'WiGS (Static, w_x=0.5)',
#         'WiGS (Static w_x=0.25)': 'WiGS (Static, w_x=0.25)', 
#         'WiGS (Time-Decay, Linear)': 'WiGS (Linear Decay)',
#         'WiGS (Time-Decay, Exponential)': 'WiGS (Exponential Decay)', 
#         'WiGS (MAB-UCB1, c=0.5)': 'WiGS (MAB, c=0.5)',
#         'WiGS (MAB-UCB1, c=2.0)': 'WiGS (MAB, c=2.0)', 
#         'WiGS (MAB-UCB1, c=5.0)': 'WiGS (MAB, c=5.0)'
#     }

#     ## Create output directory ##
#     trace_plot_dir = os.path.join(image_dir, 'trace')
#     variance_plot_dir = os.path.join(image_dir, 'variance')
#     os.makedirs(trace_plot_dir, exist_ok=True)
#     os.makedirs(variance_plot_dir, exist_ok=True)

#     ## Loop through all results files ##
#     result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.pkl')]
#     for result_file in result_files:
#         data_name = result_file.replace('_results.pkl', '')
#         print(f"\nProcessing dataset: {data_name}...")

#         # Load #
#         file_path = os.path.join(results_dir, result_file)
#         with open(file_path, 'rb') as f:
#             all_results = pickle.load(f)

#         # Plot #
#         TracePlotMean, TracePlotVariance = MeanVariancePlot(
#             RelativeError=None,
#             Colors=master_colors,
#             LegendMapping=master_legend,
#             Linestyles=master_linestyles,
#             Y_Label="Normalized RMSE (iGS = 1.0)",
#             Subtitle=f"Active Learning Performance on {data_name.upper()} Dataset",
#             TransparencyVal=0,
#             VarInput=True,
#             CriticalValue=1.96,
#             initial_train_proportion=0.16,
#             candidate_pool_proportion=0.64,
#             **all_results
#         )

#         # Save #
#         trace_plot_path = os.path.join(trace_plot_dir, f"{data_name}_TracePlot.png")
#         TracePlotMean.savefig(trace_plot_path, bbox_inches='tight', dpi=300)
#         plt.close(TracePlotMean) 

#         if TracePlotVariance:
#             variance_plot_path = os.path.join(variance_plot_dir, f"{data_name}_VariancePlot.png")
#             TracePlotVariance.savefig(variance_plot_path, bbox_inches='tight', dpi=300)
#             plt.close(TracePlotVariance)

#         print(f"Saved plots for {data_name}.")

#     print("\n--- Analysis Complete ---")
