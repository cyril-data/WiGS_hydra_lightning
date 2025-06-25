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
def TestErrorFunction(InputModel, df_Test):

    ### Get features ###
    X_test_df, y_test_series = get_features_and_target(
        df=df_Test,
        target_column_name="Y") 
    
    ### RMSE ###
    Prediction = InputModel.predict(X_test_df)
    ErrorVal = np.mean((Prediction - y_test_series)**2)
    Output = {"ErrorVal": ErrorVal.tolist()}

    ### Return ###
    return Output

            