# Summary: Calculates the loss of the test set.
# Input:
#   InputModel: The prediction model used.
#   df_Test: The test data.
#   Type: A string {"Regression", "Classification"} indicating the prediction objective.
# Output:
#   RMSE, MAE, R^2, CC

### Libraries ###
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
def TestErrorFunction(InputModel, df_Test):

    ### Get features and target ###
    X_test_df, y_test_series = get_features_and_target(
        df=df_Test,
        target_column_name="Y")

    ### Make Predictions ###
    Prediction = InputModel.predict(X_test_df)

    ### Calculate Metrics ###
    rmse_val = np.sqrt(np.mean((Prediction - y_test_series)**2))    # RMSE
    mae_val = mean_absolute_error(y_test_series, Prediction)        # MAE
    r2_val = r2_score(y_test_series, Prediction)                    # R^2
    cc_val, _ = pearsonr(y_test_series, Prediction)                 # Correlation Coefficient

    ### Output ###
    Output = {"RMSE": rmse_val, 
              "MAE": mae_val,
              "R2": r2_val,
              "CC": cc_val
              }
    return Output