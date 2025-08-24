### Packages ###
import pandas as pd

### Function ###
def get_features_and_target(df: pd.DataFrame,
                            target_column_name: str = "Y"):
    """
    Separates a DataFrame into features (X) and a target variable (y).

    Args:
        df: The dataframe.
        target_column_name: Target column name.
    """
    X_df = df.drop(columns=[target_column_name], errors='ignore')
    y_series = df[target_column_name] 

    return X_df, y_series