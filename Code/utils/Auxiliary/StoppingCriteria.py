import numpy as np
import pandas as pd 
from typing import Tuple, Optional, Dict

def qbc_stopping_criterion(
    f1_scores: np.ndarray,
    window_size: int = 3,
    performance_threshold: float = 0.01,
    min_iterations: int = 10,
    plateau_count: int = 2
) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    Implementation of stopping criterion for Query-by-Committee based active learning.
    
    Args and functionality remain the same as before...
    """
    # Previous implementation remains the same
    mean_f1_scores = np.mean(f1_scores, axis=0)
    
    if len(mean_f1_scores) < min_iterations:
        return False, None, None
    
    plateau_counter = 0
    
    for i in range(min_iterations, len(mean_f1_scores) - window_size + 1):
        window = mean_f1_scores[i:i + window_size]
        start_performance = window[0]
        end_performance = window[-1]
        relative_improvement = (end_performance - start_performance) / start_performance
        
        if relative_improvement < performance_threshold:
            plateau_counter += 1
            if plateau_counter >= plateau_count:
                return True, i, np.mean(window)
        else:
            plateau_counter = 0
    
    return False, None, None

def analyze_qbc_learning_curve(
    f1_scores: np.ndarray,
    window_size: int = 3,
    performance_threshold: float = 0.01,
    plateau_count: int = 2
) -> Dict:
    """
    Analyze QBC active learning results with specified parameters.
    
    Args:
        f1_scores: 2D array of shape (n_runs, n_iterations) containing F1 scores
        window_size: Size of window to check for performance plateau
        performance_threshold: Minimum required relative improvement
        plateau_count: Number of consecutive windows that must show plateau
    
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    
    should_stop, stop_iter, final_score = qbc_stopping_criterion(
        f1_scores,
        window_size=window_size,
        performance_threshold=performance_threshold,
        plateau_count=plateau_count
    )
    
    print(f"\nAnalysis with parameters:")
    print(f"Window size: {window_size}")
    print(f"Performance threshold: {performance_threshold}")
    print(f"Plateau count: {plateau_count}")
    print("-" * 50)
    
    if should_stop:
        print(f"Recommended stopping at iteration: {stop_iter}")
        print(f"Final mean F1 score: {final_score:.4f}")
        data_usage = (stop_iter + 1) / f1_scores.shape[1] * 100 + 20
        print(f"Percentage of data used: {data_usage:.1f}%")
        
        results = {
            'should_stop': should_stop,
            'stopping_iteration': stop_iter,
            'final_f1_score': final_score,
            'data_usage_percentage': data_usage
        }
    else:
        print("No stopping point found with these parameters")
        results = {
            'should_stop': False,
            'stopping_iteration': None,
            'final_f1_score': None,
            'data_usage_percentage': None
        }
    
    return results

def analyze_dataset_methods(
    dataset_name: str,
    passive_data: pd.DataFrame,
    random_forest_data: pd.DataFrame,
    unreal_dureal_data: pd.DataFrame,
    window_size: int = 3,
    performance_threshold: float = 0.01,
    plateau_count: int = 2,
    initial_data_percentage: float = 20.0
) -> pd.DataFrame:
    """
    Analyze all methods for a single dataset and return results in a DataFrame.
    
    Args:
        dataset_name: Name of the dataset
        passive_data: DataFrame containing passive learning results
        random_forest_data: DataFrame containing random forest results
        unreal_dureal_data: DataFrame containing UNREAL/DUREAL results
        window_size: Size of window for detecting plateaus
        performance_threshold: Threshold for improvement
        plateau_count: Number of consecutive plateaus required
        initial_data_percentage: Percentage of data already labeled at start
    
    Returns:
        DataFrame with results for all methods
    """
    # Analyze each method
    results_dict = {
        'PassiveLearning': analyze_qbc_learning_curve(
            passive_data["Error"],
            window_size=window_size,
            performance_threshold=performance_threshold,
            plateau_count=plateau_count
        ),
        'RandomForest': analyze_qbc_learning_curve(
            random_forest_data["Error"],
            window_size=window_size,
            performance_threshold=performance_threshold,
            plateau_count=plateau_count
        ),
        'UNREAL': analyze_qbc_learning_curve(
            unreal_dureal_data["Error_UNREAL"],
            window_size=window_size,
            performance_threshold=performance_threshold,
            plateau_count=plateau_count
        ),
        'DUREAL': analyze_qbc_learning_curve(
            unreal_dureal_data["Error_DUREAL"],
            window_size=window_size,
            performance_threshold=performance_threshold,
            plateau_count=plateau_count
        )
    }
    
    # Create results DataFrame
    rows = []
    for method, results in results_dict.items():
        if results['should_stop']:
            # Calculate actual data usage including initial data
            # total_iterations = len(passive_data)  # Assuming all methods have same length
            # additional_data_pct = (results['stopping_iteration'] + 1) / total_iterations * (100 - initial_data_percentage)
            # total_data_pct = initial_data_percentage + additional_data_pct
            total_data_pct = results["data_usage_percentage"]
            
            rows.append({
                'Dataset': dataset_name,
                'Method': method,
                'Stopping Iteration': results['stopping_iteration'],
                'Final F1': results['final_f1_score'],
                'Data_Usage_Percent (math wrong)': total_data_pct
            })
        else:
            rows.append({
                'Dataset': dataset_name,
                'Method': method,
                'Stopping Iteration': np.nan,
                'Final F1': np.nan,
                'Data_Usage_Percent (math wrong)': np.nan
            })
    
    return pd.DataFrame(rows)

def create_latex_table(results_df: pd.DataFrame) -> str:
    """
    Create a LaTeX table from results DataFrame.
    """
    latex_table = """\\begin{table}[t]
\\caption{Active Learning Performance Comparison (Starting with 20\\% Labeled Data)}
\\label{tab:al-comparison}
\\centering
\\begin{tabular}{lcccc}
\\toprule
Dataset & Method & Stopping Iter. & Final F1 & Total Data (\\%) \\\\
\\midrule
"""
    
    current_dataset = ""
    for _, row in results_df.iterrows():
        if row['Dataset'] != current_dataset:
            if current_dataset != "":
                latex_table += "\\midrule\n"
            current_dataset = row['Dataset']
        
        dataset_name = row['Dataset'] if current_dataset != row['Dataset'] else ''
        stopping_iter = f"{int(row['Stopping Iteration'])}" if not np.isnan(row['Stopping Iteration']) else '-'
        final_f1 = f"{row['Final F1']:.3f}" if not np.isnan(row['Final F1']) else '-'
        data_usage = f"{row['Data_Usage_Percent (math wrong)']:.1f}" if not np.isnan(row['Data_Usage_Percent (math wrong)']) else '-'
        
        latex_table += f"{dataset_name} & {row['Method']} & {stopping_iter} & {final_f1} & {data_usage} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex_table
