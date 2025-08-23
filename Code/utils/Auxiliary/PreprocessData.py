### Import Packages ###
import os
import sys
import pickle
import kagglehub
import pandas as pd
import numpy as np
from pmlb import fetch_data
from sklearn.preprocessing import StandardScaler

### Helper Function for Preprocessing ###
def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to a dataframe:
    1. Separates features (X) and target (Y).
    2. One-hot encodes categorical features.
    3. Standardizes all features to mean=0, std=1.
    4. Recombines features and target.
    """
    if 'Y' not in df.columns:
        raise ValueError("DataFrame must have a target column named 'Y'.")
    
    X = df.drop(columns=['Y'])
    y = df['Y']
    
    # Identify categorical columns to ensure correct type before encoding
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    
    # Standardize all features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_encoded)
    
    # Convert scaled features back to a DataFrame
    X_scaled = pd.DataFrame(X_scaled_array, columns=X_encoded.columns, index=X_encoded.index)
    
    # Recombine and return
    return pd.concat([y, X_scaled], axis=1)

### Main Preprocessing and Saving Function ###
def preprocess_and_save_all():
    """
    Loads all datasets, applies the full preprocessing pipeline, and saves them.
    """
    print("--- Starting Data Preprocessing (including encoding and standardization) ---")
    
    # --- Setup Paths ---
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    datasets_to_save = {}

    print("Loading and processing all datasets...")

    # --- Load, Process, and Collect All Datasets ---
    # Each dataset is loaded raw, then passed to the helper function
    
    # 1. Concrete Compressive Strength
    concrete_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    df_raw = pd.read_excel(concrete_url).rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'Y'})
    datasets_to_save['concrete_4'] = _preprocess_dataframe(df_raw)

    # 2. Concrete Slump (CS, Flow, Slump)
    slump_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
    df_slump_base = pd.read_csv(slump_url)
    df_cs_raw = df_slump_base.drop(columns=['FLOW(cm)', 'SLUMP(cm)']).rename(columns={'Compressive Strength (28-day)(Mpa)': 'Y'})
    df_flow_raw = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'SLUMP(cm)']).rename(columns={'FLOW(cm)': 'Y'})
    df_slump_raw = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'FLOW(cm)']).rename(columns={'SLUMP(cm)': 'Y'})
    datasets_to_save['concrete_cs'] = _preprocess_dataframe(df_cs_raw)
    datasets_to_save['concrete_flow'] = _preprocess_dataframe(df_flow_raw)
    datasets_to_save['concrete_slump'] = _preprocess_dataframe(df_slump_raw)

    # 3. Yacht Hydrodynamics
    yacht_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    yacht_columns = ['longitudinal_pos', 'prismatic_coeff', 'length_displacement_ratio', 'beam_draught_ratio', 'length_beam_ratio', 'froude_number', 'Y']
    df_raw = pd.read_csv(yacht_url, sep=r'\s+', header=None, names=yacht_columns)
    datasets_to_save['yacht'] = _preprocess_dataframe(df_raw)

    # 4. Housing
    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df_raw = pd.read_csv(housing_url, sep=r'\s+', header=None, names=housing_columns)
    df_raw = df_raw.rename(columns={'MEDV': 'Y'})
    datasets_to_save['housing'] = _preprocess_dataframe(df_raw)
    
    # 5. Auto MPG
    mpg_column_names = ['Y', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    mpg_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    df_raw = pd.read_csv(mpg_url, sep=r'\s+', header=None, names=mpg_column_names, na_values='?')
    del df_raw['car_name']
    df_raw.dropna(inplace=True)
    datasets_to_save['mpg'] = _preprocess_dataframe(df_raw)

    # 6. Wine (Red and White)
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df_red_raw = pd.read_csv(url_red, sep=';').rename(columns={'quality': 'Y'})
    datasets_to_save['wine_red'] = _preprocess_dataframe(df_red_raw)
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df_white_raw = pd.read_csv(url_white, sep=';').rename(columns={'quality': 'Y'})
    datasets_to_save['wine_white'] = _preprocess_dataframe(df_white_raw)
    
    # 7. CPS
    cps_url = 'http://lib.stat.cmu.edu/datasets/CPS_85_Wages'
    df_raw = pd.read_csv(cps_url, sep=r'\s+', skiprows=27, header=None)
    df_raw.columns = ["EDUCATION", "SOUTH", "SEX", "EXPERIENCE", "UNION", "WAGE", "AGE", "RACE", "OCCUPATION", "SECTOR", "MARR"]
    df_raw = df_raw.rename(columns={'WAGE': 'Y'})
    datasets_to_save['cps'] = _preprocess_dataframe(df_raw)

    # 8. pmlb datasets (NO2 and PM10)
    df_pm10_raw = fetch_data('529_pollen', return_X_y=False).rename(columns={'target': 'Y'})
    datasets_to_save['pm10'] = _preprocess_dataframe(df_pm10_raw)
    df_no2_raw = fetch_data('560_bodyfat', return_X_y=False).rename(columns={'target': 'Y'})
    datasets_to_save['no2'] = _preprocess_dataframe(df_no2_raw)

    # 9. QSAR Aquatic Toxicity
    qsar_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv'
    df_raw = pd.read_csv(qsar_url, sep=';', header=None)
    df_raw.columns = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI','GATS1p', 'nN', 'C040', 'Y']
    datasets_to_save['qsar'] = _preprocess_dataframe(df_raw)

    # 10. Kaggle datasets
    try:
        download_path_bf = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")
        df_raw = pd.read_csv(os.path.join(download_path_bf, 'bodyfat.csv'))
        if 'Density' in df_raw.columns: del df_raw['Density']
        df_raw = df_raw.rename(columns={'BodyFat': 'Y'})
        datasets_to_save['bodyfat'] = _preprocess_dataframe(df_raw)

        download_path_beer = kagglehub.dataset_download("dongeorge/beer-consumption-sao-paulo")
        df_raw = pd.read_csv(os.path.join(download_path_beer, 'Consumo_cerveja.csv'), decimal=',')
        df_raw.dropna(inplace=True)
        df_raw.columns = ['Date', 'Temp_Avg_C', 'Temp_Min_C', 'Temp_Max_C', 'Precipitation_mm', 'Weekend', 'Y']
        del df_raw['Date']
        for col in df_raw.columns:
            if col != 'Y': df_raw[col] = df_raw[col].astype(float)
        datasets_to_save['beer'] = _preprocess_dataframe(df_raw)
    except Exception as e:
        print(f"\n--- KAGGLE ERROR: {e} ---")

    # 11. Burbridge Dataset
    df_dgp_correct = generate_burbidge_data(delta=0.0, sigma_epsilon=0.3, seed=42)
    datasets_to_save['dgp_correct'] = _preprocess_dataframe(df_dgp_correct)
    df_dgp_misspecified = generate_burbidge_data(delta=0.05, sigma_epsilon=0.3, seed=42)
    datasets_to_save['dgp_misspecified'] = _preprocess_dataframe(df_dgp_misspecified)
    df_dgp_low_noise = generate_burbidge_data(delta=0.05, sigma_epsilon=0.1, seed=42)
    datasets_to_save['dgp_low_noise'] = _preprocess_dataframe(df_dgp_low_noise)

    # --- Save all successfully processed datasets ---
    print("\nSaving all processed datasets...")
    for name, dataframe in datasets_to_save.items():
        file_path = os.path.join(save_path, f"{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(dataframe, file)
        print(f"  > Successfully saved: {name}.pkl")
    
    print("\n--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    preprocess_and_save_all()