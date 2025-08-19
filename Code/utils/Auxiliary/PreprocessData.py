### Import Packages ###
import os
import pickle
import kagglehub
import pandas as pd
import numpy as np
from pmlb import fetch_data

def PreprocessData():
    print("--- Starting Data Preprocessing ---")
    
    # --- Setup Save Directory (Robust Pathing) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    datasets_to_save = {}

    # --- Load, Process, and Collect All Datasets ---
    print("Processing UCI and StatLib datasets...")

    # Concrete Compression
    concrete_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    df_concrete = pd.read_excel(concrete_url)
    datasets_to_save['concrete_4'] = df_concrete.rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'Y'})

    # Concrete Slump (CS, Flow, Slump)
    slump_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
    df_slump_base = pd.read_csv(slump_url)
    df_concrete_cs = df_slump_base.drop(columns=['FLOW(cm)', 'SLUMP(cm)']).rename(columns={'Compressive Strength (28-day)(Mpa)': 'Y'})
    df_concrete_flow = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'SLUMP(cm)']).rename(columns={'FLOW(cm)': 'Y'})
    df_concrete_slump = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'FLOW(cm)']).rename(columns={'SLUMP(cm)': 'Y'})
    datasets_to_save.update({'concrete_cs': df_concrete_cs, 'concrete_flow': df_concrete_flow, 'concrete_slump': df_concrete_slump})

    # Yacht
    yacht_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    yacht_columns = ['longitudinal_pos', 'prismatic_coeff', 'length_displacement_ratio', 'beam_draught_ratio', 'length_beam_ratio', 'froude_number', 'Y']
    datasets_to_save['yacht'] = pd.read_csv(yacht_url, sep=r'\s+', header=None, names=yacht_columns)

    # Housing
    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df_housing = pd.read_csv(housing_url, sep=r'\s+', header=None, names=housing_columns)
    datasets_to_save['housing'] = df_housing.rename(columns={'MEDV': 'Y'})
    
    # Auto MPG
    mpg_column_names = ['Y', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    mpg_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    df_auto_mpg = pd.read_csv(mpg_url, sep=r'\s+', header=None, names=mpg_column_names, na_values='?')
    del df_auto_mpg['car_name']
    df_auto_mpg.dropna(inplace=True)
    datasets_to_save['mpg'] = df_auto_mpg

    # Wine (Red and White)
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df_wine_red = pd.read_csv(url_red, sep=';').rename(columns={'quality': 'Y'})
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df_wine_white = pd.read_csv(url_white, sep=';').rename(columns={'quality': 'Y'})
    datasets_to_save.update({'wine_red': df_wine_red, 'wine_white': df_wine_white})
    
    # CPS
    cps_url = 'http://lib.stat.cmu.edu/datasets/CPS_85_Wages'
    df_cps = pd.read_csv(cps_url, sep=r'\s+', skiprows=27).rename(columns={'WAGE': 'Y'})
    datasets_to_save['cps'] = df_cps

    # pmlb datasets
    print("Processing pmlb datasets...")
    df_pm10 = fetch_data('529_pollen', return_X_y=False).rename(columns={'target': 'Y'})
    df_no2 = fetch_data('560_bodyfat', return_X_y=False).rename(columns={'target': 'Y'})
    datasets_to_save.update({'pm10': df_pm10, 'no2': df_no2})

    # QSAR
    qsar_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv'
    df_qsar = pd.read_csv(qsar_url, sep=';', header=None)
    df_qsar.columns = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI','GATS1p', 'nN', 'C040', 'Y']
    datasets_to_save['qsar'] = df_qsar

    # Kaggle datasets
    print("Processing Kaggle Hub datasets...")
    download_path_bf = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")
    df_bodyfat = pd.read_csv(os.path.join(download_path_bf, 'bodyfat.csv'))
    if 'Density' in df_bodyfat.columns: del df_bodyfat['Density']
    datasets_to_save['bodyfat'] = df_bodyfat.rename(columns={'BodyFat': 'Y'})

    download_path_beer = kagglehub.dataset_download("dongeorge/beer-consumption-sao-paulo")
    df_beer = pd.read_csv(os.path.join(download_path_beer, 'Consumo_cerveja.csv'), decimal=',')
    df_beer.dropna(inplace=True)
    df_beer.columns = ['Date', 'Temp_Avg_C', 'Temp_Min_C', 'Temp_Max_C', 'Precipitation_mm', 'Weekend', 'Y']
    del df_beer['Date']
    datasets_to_save['beer'] = df_beer
    
    # --- Save all datasets ---
    print("\nSaving all processed datasets...")
    for name, dataframe in datasets_to_save.items():
        file_path = os.path.join(save_path, f"{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(dataframe, file)
        print(f"  > Successfully saved: {name}.pkl")
    
    print("\n--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    PreprocessData()