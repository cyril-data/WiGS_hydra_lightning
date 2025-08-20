### Libraries ###
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
def PaperTestErrorMetrics(InputModel, df_Train, df_Candidate):

    if df_Candidate.empty:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "CC": np.nan}
    
    X_cand, y_cand_true = get_features_and_target(df=df_Candidate, target_column_name="Y")
    y_cand_pred = InputModel.predict(X_cand)

    _, y_train_true = get_features_and_target(df=df_Train, target_column_name="Y")
    
    pool_true = pd.concat([y_train_true, y_cand_true], ignore_index=True)
    pool_pred = pd.concat([y_train_true, pd.Series(y_cand_pred, index=y_cand_true.index)], ignore_index=True)
    
    rmse_val = np.sqrt(np.mean((pool_pred - pool_true)**2))
    mae_val = mean_absolute_error(pool_true, pool_pred)
    r2_val = r2_score(pool_true, pool_pred)
    cc_val, _ = pearsonr(pool_true, pool_pred)

    Output = {"RMSE": rmse_val, "MAE": mae_val, "R2": r2_val, "CC": cc_val}
    return Output