### Packages ###
import pandas as pd


### Function ###
def get_features_and_target(df: pd.DataFrame, target_column_name: str = "Y", y_size: int = 1):
    """
    Separates a DataFrame into features (X) and a target variable (y).

    Args:
        df: The dataframe.
        target_column_name: Target column name.
    """
    if target_column_name == None:
        X_df = df
        y_series = None
        X_df_iloc = df
    else:
        X_df = df.drop(columns=[target_column_name], errors="ignore")
        y_series = df[target_column_name]
        X_df_iloc = df.iloc[:, y_size:]
        y_series_iloc = df.iloc[:, :y_size]
        # print("y_series iloc", y_series)

    # print("X_df", X_df)
    # X_df_ysize = df.iloc[:, y_size:]
    # print("X_df_ysize", X_df_ysize)

    # # if y_size == None:
    # #     y_series = None
    # # else:
    # #     y_series = df.iloc[:, :y_size]

    # print("X_df", X_df.columns)
    # print("X_df_ysize y_size", y_size, X_df_ysize.columns)

    diff = X_df.compare(X_df_iloc)  # Affiche les différences

    if y_series is not None:
        # print("diff y_series", type(y_series), y_series.isnull().to_list())
        # print("y_series_iloc", type(y_series_iloc), y_series.isnull().to_list())
        diff = y_series.to_frame().compare(y_series_iloc)  # Affiche les différences
        # print("diff y_series", diff)

    X_df = X_df_iloc

    return X_df, y_series_iloc if y_series is not None else y_series


# ### Function ###
# def get_features_and_target(df: pd.DataFrame, y_size: int):
#     """
#     Separates a DataFrame into features (X) and a target variable (y).

#     Args:
#         df: The dataframe.
#         target_column_name: Target column name.
#     """

#     X_df = df.iloc[:, y_size:]
#     y_series = df.iloc[:, :y_size]

#     return X_df, y_series
