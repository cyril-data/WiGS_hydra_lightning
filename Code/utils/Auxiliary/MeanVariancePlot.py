# Summary: Creates a plot for the average error and the average variance of each active learning strategy averaged 
#          across simulations.
# Input:
#   Subtitle: A string subtitle for the plot.
#   TransparencyVal: A float value indicating the transparency of the confidence interval.
#   CriticalValue: The critical value of the confidence interval.
#   RelativeError: A string whose value is one of the names of the input SimulationErrorResults indicating whether
#                  to make the plot relative to this value. The graph's line of the input for RelativeError will be 
#                  1 to form a baseline across the simulation with all other errors and variance divided by the baseline.
#   Colors: A dictionary of colors for each active learning strategy in SimulationErrorResults. 
#   SimulationErrorResults: The error rates across iteration for each active learning strategy.
# Output: Two plots MeanPlot and VariancePlot representing the mean and variance of the active learning strategies 
#         averaged across simulations.

### Import packages ###
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle = None,
                     TransparencyVal = 0.2,
                     CriticalValue = 1.96,
                     RelativeError = None,
                     Colors= None, 
                     Linestyles = None,
                     Markerstyles = None,
                     xlim = None,
                     Y_Label = None,
                     VarInput = False,
                     FigSize = (10,4),
                     LegendMapping = None,
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}
    StdErrorVector ={}
    StdErrorVarianceVector = {}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis=0)
        VarianceVector[Label] = np.var(Results, axis=0)
        StdErrorVector[Label] = np.std(Results, axis=0) / np.sqrt(Results.shape[0])
        
        # Compute CI for variance
        n = Results.shape[0]  # Number of samples
        lower_chi2 = chi2.ppf(0.025, df=n-1)  # Lower bound of Chi-square
        upper_chi2 = chi2.ppf(0.975, df=n-1)  # Upper bound of Chi-square
        StdErrorVarianceVector[Label] = {
            "lower": (n-1) * VarianceVector[Label] / upper_chi2,
            "upper": (n-1) * VarianceVector[Label] / lower_chi2
        }

    ### Normalize to Relative Error if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = "Mean Error relative to " + RelativeError
            BaselineMean = MeanVector[RelativeError]
            BaselineVariance = VarianceVector[RelativeError]
            
            for Label in MeanVector:
                MeanVector[Label] = pd.Series((MeanVector[Label].values - BaselineMean.values) / BaselineMean.values, 
                                            index=MeanVector[Label].index)
                StdErrorVector[Label] = pd.Series(StdErrorVector[Label].values / BaselineMean.values, 
                                                index=StdErrorVector[Label].index)
                VarianceVector[Label] = pd.Series(VarianceVector[Label].values / BaselineVariance.values, 
                                                index=VarianceVector[Label].index)
        else:
            raise ValueError(f"RelativeError='{RelativeError}' not found in provided results.")

    ### Mean Plot ###
    plt.figure(figsize=FigSize)
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        x = 20 + (np.arange(len(MeanValues)) / len(MeanValues)) * 80  # Start at 20% and go to 100%
        color = Colors.get(Label, None) if Colors else None 
        linestyle = Linestyles.get(Label, ':') if Linestyles else ':'
        markerstyle = Markerstyles.get(Label, 'o') if Markerstyles else 'o'
        legend_label = LegendMapping[Label] if LegendMapping and Label in LegendMapping else Label
        plt.plot(x, MeanValues, label=legend_label, color=color, linestyle=linestyle)
        # plt.plot(x, MeanValues, label=legend_label, color=color, linestyle=linestyle, marker = markerstyle, markersize=2)
        plt.fill_between(x, MeanValues - CriticalValue * StdErrorValues, 
                         MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal, color=color)
    # plt.suptitle("Active Learning Mean Error Plot")
    plt.xlabel("Percent of labelled observations")
    plt.ylabel(Y_Label)
    plt.title(Subtitle, fontsize=9)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='upper right')
    if type(xlim) == list:
        plt.xlim(xlim)
    else: 
        pass
    MeanPlot = plt.gcf()

    # Variance Plot
    if VarInput:
        plt.figure(figsize= FigSize)
        for Label, VarianceValues in VarianceVector.items():
            x = 20 + (np.arange(len(VarianceValues)) / len(VarianceValues)) * 80  # Start at 20% and go to 100%
            color = Colors.get(Label, None) if Colors else None
            linestyle = Linestyles.get(Label, '-') if Linestyles else '-'
            markerstyle = Markerstyles.get(Label, 'o') if Markerstyles else 'o'
            legend_label = LegendMapping[Label] if LegendMapping and Label in LegendMapping else Label
            plt.plot(x, VarianceValues, label=legend_label, color=color, linestyle=linestyle)
            # plt.plot(x, VarianceValues, label=legend_label, color=color, linestyle=linestyle, marker = markerstyle, markersize=2)
            lower_bound = StdErrorVarianceVector[Label]["lower"]
            upper_bound = StdErrorVarianceVector[Label]["upper"]
            plt.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)
        # plt.suptitle("Active Learning Variance of Error Plot")
        plt.xlabel("Percent of labelled observations")
        plt.ylabel("Variance of " + Y_Label)
        plt.title(Subtitle, fontsize = 9)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.legend(loc='upper right')
        if type(xlim) == list:
            plt.xlim(xlim)
        else: 
            pass
        VariancePlot = plt.gcf()
    else:
        VariancePlot = None

    ### Return ###
    if VarInput:
        return MeanPlot, VariancePlot
    else:
        return MeanPlot

