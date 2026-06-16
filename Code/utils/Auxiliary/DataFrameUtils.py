### Packages ###
import pandas as pd


### Function ###
def get_features_and_target(df: pd.DataFrame, y_size: int = 1):
    """
    Separates a DataFrame into features (X) and a target variable (y).

    Args:
        df: The dataframe.
        target_column_name: Target column name.
    """
    if y_size == None:
        y_series_iloc = None
        X_df_iloc = df
    else:
        if y_size == 1 and df.columns.to_list()[0] == "Y":  # database model with "Y" first col
            X_df_iloc = df.iloc[:, y_size:]
            y_series_iloc = df.iloc[:, :y_size]
        else:  # database hydra with targets = last colone :     X[:x_size] and y[x_size:]
            x_size = df.shape[1] - y_size
            X_df_iloc = df.iloc[:, :x_size]
            y_series_iloc = df.iloc[:, x_size:]

    return X_df_iloc, y_series_iloc if y_size is not None else y_series_iloc
