### Import packages ###
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

### Function ###
def MeanVariancePlot(Subtitle=None,
                     TransparencyVal=0.2,
                     CriticalValue=1.96,
                     RelativeError=None,
                     Colors=None,
                     Linestyles=None,
                     Markerstyles=None,
                     xlim=None,
                     Y_Label=None,
                     VarInput=False,
                     FigSize=(10, 4),
                     LegendMapping=None,
                     initial_train_proportion=0.16, # Default based on 0.2 test/0.8 candidate split
                     candidate_pool_proportion=0.64, # Default based on 0.2 test/0.8 candidate split
                     **SimulationErrorResults):

    ### Set Up ###
    MeanVector = {}
    VarianceVector = {}
    StdErrorVector = {}
    StdErrorVarianceVector = {}

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():
        MeanVector[Label] = np.mean(Results, axis=1)
        # MeanVector[Label] = np.median(Results, axis=1)
        VarianceVector[Label] = np.var(Results, axis=1)
        
        n_simulations = Results.shape[1]
        StdErrorVector[Label] = np.std(Results, axis=1) / np.sqrt(n_simulations)
        
        # Compute CI for variance using corrected n
        lower_chi2 = chi2.ppf(0.025, df=n_simulations - 1)
        upper_chi2 = chi2.ppf(0.975, df=n_simulations - 1)
        StdErrorVarianceVector[Label] = {
            "lower": (n_simulations - 1) * VarianceVector[Label] / upper_chi2,
            "upper": (n_simulations - 1) * VarianceVector[Label] / lower_chi2
        }

    ### Normalize to Relative Error if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = f"Normalized Error (Baseline: {RelativeError}=1.0)"
            BaselineMean = MeanVector[RelativeError].copy()
            BaselineVariance = VarianceVector[RelativeError].copy()
            
            for Label in MeanVector:
                MeanVector[Label] = MeanVector[Label] / BaselineMean
                StdErrorVector[Label] = StdErrorVector[Label] / BaselineMean
                VarianceVector[Label] = VarianceVector[Label] / BaselineVariance
        else:
            raise ValueError(f"RelativeError='{RelativeError}' not found in provided results.")

    ### Mean Plot ###
    plt.figure(figsize=FigSize)
    for Label, MeanValues in MeanVector.items():
        StdErrorValues = StdErrorVector[Label]
        
        num_iterations = len(MeanValues)
        if num_iterations > 1:
            iterations_array = np.arange(num_iterations)
            x = (initial_train_proportion + (iterations_array / (num_iterations - 1)) * candidate_pool_proportion) * 100
        else: 
            x = [initial_train_proportion * 100]

        color = Colors.get(Label, None) if Colors else None
        linestyle = Linestyles.get(Label, ':') if Linestyles else ':'
        markerstyle = Markerstyles.get(Label, 'o') if Markerstyles else 'o'
        legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
        
        plt.plot(x, MeanValues, label=legend_label, color=color, linestyle=linestyle)
        plt.fill_between(x, MeanValues - CriticalValue * StdErrorValues,
                         MeanValues + CriticalValue * StdErrorValues, alpha=TransparencyVal, color=color)
        
    plt.xlabel("Percent of Total Data Labeled for Training")
    plt.ylabel(Y_Label)
    plt.title(Subtitle, fontsize=9)
    plt.legend(loc='upper right')
    if isinstance(xlim, list):
        plt.xlim(xlim)
    
    MeanPlot = plt.gcf()
    plt.show() 

    # Variance Plot
    if VarInput:
        plt.figure(figsize=FigSize)
        for Label, VarianceValues in VarianceVector.items():
            num_iterations = len(VarianceValues)
            if num_iterations > 1:
                iterations_array = np.arange(num_iterations)
                x = (initial_train_proportion + (iterations_array / (num_iterations - 1)) * candidate_pool_proportion) * 100
            else:
                x = [initial_train_proportion * 100]

            color = Colors.get(Label, None) if Colors else None
            linestyle = Linestyles.get(Label, '-') if Linestyles else '-'
            markerstyle = Markerstyles.get(Label, 'o') if Markerstyles else 'o'
            legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label
            
            plt.plot(x, VarianceValues, label=legend_label, color=color, linestyle=linestyle)
            
            lower_bound = StdErrorVarianceVector[Label]["lower"]
            upper_bound = StdErrorVarianceVector[Label]["upper"]
            plt.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)
            
        plt.xlabel("Percent of Total Data Labeled for Training")
        plt.ylabel("Variance of " + (Y_Label if Y_Label else "Error"))
        plt.title(Subtitle, fontsize=9)
        plt.legend(loc='upper right')
        if isinstance(xlim, list):
            plt.xlim(xlim)
        
        VariancePlot = plt.gcf()
        plt.show() 
    else:
        VariancePlot = None

    ### Return ###
    if VarInput:
        return MeanPlot, VariancePlot
    else:
        return MeanPlot