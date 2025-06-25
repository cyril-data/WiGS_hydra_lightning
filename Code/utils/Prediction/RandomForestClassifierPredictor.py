### Libraries ###
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierPredictor:

    ### Initialize Model ###
    def __init__(self, Seed: int, n_estimators: int = 100, **kwargs):
        self.Seed = Seed
        self.n_estimators = n_estimators
        self.model = None 
        np.random.seed(self.Seed) 

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.Seed)
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_data_df)

    ### Get prediction probabilities ###
    def predict_proba(self, X_data_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X_data_df)

    ### Raw Ensemble ###
    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        
        ## Set Up ##
        individual_tree_preds = []   # Initialize
        X_data_np = X_data_df.values # Convert X_data_df to a NumPy array 
        
        ## Store raw predictions ## 
        for tree_estimator in self.model.estimators_:
            individual_tree_preds.append(tree_estimator.predict(X_data_np))
        
        ## Get ensemsble prediction s##
        ensemble_predictions_df = pd.DataFrame(np.vstack(individual_tree_preds)).T 
        ensemble_predictions_df.columns = [f"tree_{i}" for i in range(ensemble_predictions_df.shape[1])]
        ensemble_predictions_df.index = X_data_df.index 
        return ensemble_predictions_df