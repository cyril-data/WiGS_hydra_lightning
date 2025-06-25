# Summary: Calculates the loss of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:

### Libraries ###
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
def TestErrorFunction(InputModel, df_Test, Type, auxiliary_columns=None):

    ### Get features ###
    X_test_df, y_test_series = get_features_and_target(
        df=df_Test,
        target_column_name="Y",
        auxiliary_columns=auxiliary_columns 
    )
    X_test_np = X_test_df.values
    y_test_np = y_test_series.values 
    
    ### RMSE ###
    if Type == "Regression":
        Prediction = InputModel.predict(X_test_df)
        ErrorVal = np.mean((Prediction - y_test_series)**2)
        Output = {"ErrorVal": ErrorVal.tolist()}

    ### Classification Error ###
    elif Type == "Classification":
        
        ### BALD or TreefarmsLFRPredictor ###
        if hasattr(InputModel, 'predict_proba_K'):


            ### BALD or TreeFARMS ###
            K_for_test_eval = 100 
            log_probs_N_K_C_test = InputModel.predict_proba_K(X_test_np, K_for_test_eval)       # Pass the already correctly filtered X_test_np
            probs_N_K_C_test = torch.exp(log_probs_N_K_C_test)                                  # Convert log-probabilities to probabilities
            mean_probs_N_C_test = torch.mean(probs_N_K_C_test, dim=1)                           # Average probabilities across K samples
            ensemble_prediction_test = torch.argmax(mean_probs_N_C_test, dim=1).cpu().numpy()   # Get the predictions
            ErrorVal = float(f1_score(y_test_np, ensemble_prediction_test, average='micro'))    # Calculate F1 score
            Output = {"ErrorVal": ErrorVal}                                                     # Output
            
            ### Store Trees for TreeFARMS ##
            if hasattr(InputModel, 'get_tree_counts'): 
                 tree_counts = InputModel.get_tree_counts() 
                 Output["AllTreeCount"] = tree_counts["AllTreeCount"]
                 Output["UniqueTreeCount"] = tree_counts["UniqueTreeCount"]
        
        ### Other Models ###
        else:
            Prediction = InputModel.predict(X_test_df) 
            ErrorVal = float(f1_score(y_test_np, Prediction, average='micro'))
            Output = {"ErrorVal": ErrorVal}

    ### Return ###
    return Output

            