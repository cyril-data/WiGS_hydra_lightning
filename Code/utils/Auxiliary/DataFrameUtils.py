import pandas as pd

def get_features_and_target(df: pd.DataFrame, 
                            target_column_name: str = "Y",
                            auxiliary_columns: list = None):
    if auxiliary_columns is None:
        auxiliary_columns = []

    # Columns to exclude from features
    cols_to_exclude = [target_column_name] + auxiliary_columns
    
    # Filter out columns that actually exist in the DataFrame
    existing_cols_to_exclude = [col for col in cols_to_exclude if col in df.columns]

    X_df = df.drop(columns=existing_cols_to_exclude, errors='ignore')
    y_series = df[target_column_name] # Assuming target column always exists

    return X_df, y_series