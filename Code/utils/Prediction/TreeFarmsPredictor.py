# Summary: Initializes and fits a treefarms model using the standard TreeFarms library.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThresholdType: Type of Rashomon threshold calculation ("Adder" or "Multiplier").
#   RashomonThreshold: A float indicating the Rashomon threshold
# Output:
# treeFarmsModel: A beautiful and amazing treefarms model yay.

### Libraries ###
from treeFarms.treefarms.model.treefarms import TREEFARMS
import pandas as pd
import numpy as np
from scipy import stats 
import torch


### Predict Single Tree ##
def _predict_single_tree(tree_model, X_df: pd.DataFrame):
    predictions = []
    data = X_df.values 
    for i in range(data.shape[0]):
        prediction, _ = tree_model.classify(data[i, :])
        predictions.append(prediction)
    return pd.Series(predictions, index=X_df.index)

### Score Single Tree ##
def _score_single_tree(tree_model, X: pd.DataFrame, y: pd.Series):
    return (_predict_single_tree(tree_model, X) == y).mean()

class TreeFarmsPredictor:

    ### Initialize Model ###
    def __init__(self, regularization: float, RashomonThreshold: float, 
                 RashomonThresholdType: str = "Adder", Seed: int = None, **kwargs):
        self.regularization = regularization
        self.RashomonThreshold = RashomonThreshold
        self.RashomonThresholdType = RashomonThresholdType
        self.Seed = Seed 

        self.model = None 
        self.all_trees = [] 
        self.X_train_columns = None 
        self.y_train_classes = None
 

    ### Fit model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        self.X_train_columns = X_train_df.columns 
        self.y_train_classes = y_train_series 
        
        config = {"regularization": self.regularization}

        if self.RashomonThresholdType == "Adder":
            config["rashomon_bound_adder"] = self.RashomonThreshold
        elif self.RashomonThresholdType == "Multiplier":
            config["rashomon_bound_multiplier"] = self.RashomonThreshold
        else:
            raise ValueError(f"Unknown RashomonThresholdType: {self.RashomonThresholdType}")

        self.model = TREEFARMS(config)
        self.model.fit(X_train_df, y_train_series)
        self.all_trees = [self.model[i] for i in range(self.model.get_tree_count())]


    ### Helper function to predict with a single tree from the ensemble ###
    def _predict_single_tree(self, tree_model, X_df: pd.DataFrame):
        predictions = []
        data = X_df.values
        for i in range(data.shape[0]):
            prediction, _ = tree_model.classify(data[i, :])
            predictions.append(prediction)
        return pd.Series(predictions, index=X_df.index)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        if not self.all_trees: 
            return np.zeros(X_data_df.shape[0]) 

        # Get predictions of each tree #
        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)
        
        # Get ensemble prediction #
        mode_predictions = stats.mode(ensemble_predictions_df, axis=1)[0].squeeze()
        return mode_predictions.astype(int) 

    ### Get prediction probabilities ###
    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int = None) -> torch.Tensor:

        # Convert numpy array back to DataFrame for ensemble prediction
        X_data_df = pd.DataFrame(X_data_np, columns=self.X_train_columns)

        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)

        num_samples = ensemble_predictions_df.shape[0]
        num_trees_in_ensemble = ensemble_predictions_df.shape[1]
        
        # Determine unique classes from training data
        unique_classes = np.unique(self.y_train_classes) 
        num_classes = len(unique_classes)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        
        # Initialize tensor for log probabilities
        log_probs_N_K_C = torch.full((num_samples, num_trees_in_ensemble, num_classes), -float('inf'), dtype=torch.float32)

        for sample_idx in range(num_samples):
            for tree_idx in range(num_trees_in_ensemble):
                predicted_class_label = ensemble_predictions_df.iloc[sample_idx, tree_idx]
                mapped_class_index = class_mapping.get(predicted_class_label)
                if mapped_class_index is not None: 
                    log_probs_N_K_C[sample_idx, tree_idx, mapped_class_index] = 0.0 

        # Output #
        return log_probs_N_K_C

    ### Get duplicate and unique count ##
    def get_tree_counts(self) -> dict:
        all_tree_count = self.model.get_tree_count() if self.model else 0
        unique_tree_count = self.model.get_unique_tree_count() if self.model and hasattr(self.model, 'get_unique_tree_count') else all_tree_count # Assuming TREEFARMS might have this method
        
        return {"AllTreeCount": all_tree_count, "UniqueTreeCount": unique_tree_count}
    
    ### Get and store prediction from for each tree ###
    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        ensemble_predictions_list = [self._predict_single_tree(tree, X_data_df) for tree in self.all_trees]
        ensemble_predictions_df = pd.concat(ensemble_predictions_list, axis=1)
        ensemble_predictions_df.columns = [f"tree_{i}" for i in range(ensemble_predictions_df.shape[1])]

        # ### Save ###
        # import os
        # filename = "/Users/simondn/Downloads/TreeFarmsBase/"
        # base_name, ext = os.path.splitext(filename)
        # counter = 0
        # new_filename = filename

        # while os.path.exists(new_filename):
        #     counter += 1
        #     new_filename = f"{base_name}_{counter}{ext}"

        # ensemble_predictions_df.to_csv(new_filename, index=False)
        # print(f"File saved as: {new_filename}")
        # ######
        return ensemble_predictions_df