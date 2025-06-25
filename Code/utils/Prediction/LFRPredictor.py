### Libraries ###
from treefarms import TREEFARMS as tf
import pandas as pd
import numpy as np
import torch
from scipy import stats
import warnings

DEFAULT_STATIC_CONFIG = {
    "depth_budget": 3,
    "rashomon_ignore_trivial_extensions": True,
    "verbose": False
}

### Predict Single Tree ##
def _predict_single_tree(tree_model, X_np: np.ndarray):
    """
    Predicts labels for a NumPy array of data using a single TreeFarms tree model.
    """
    predictions = []
    for i in range(X_np.shape[0]):
        prediction, _ = tree_model.classify(X_np[i, :])
        predictions.append(prediction)
    return predictions 

### Score Single Tree ##
def _score_single_tree(tree_model, X_np: np.ndarray, y_np: np.ndarray):
    """
    Calculates the accuracy of a single TreeFarms tree model on NumPy arrays.
    """
    return (np.array(_predict_single_tree(tree_model, X_np)) == y_np).mean()

### LFR Predictor ###
class LFRPredictor:

    ### Initialize Model ###
    def __init__(self,
                 regularization: float,
                 RashomonThreshold: float,      # max_epsilon for tuning
                 RashomonThresholdType: str = "Adder",
                 auto_tune_epsilon: bool = True, # 
                 **kwargs):
        self.regularization = regularization
        self.full_epsilon = 2*RashomonThreshold # full_epsilon is the last threshold used in the last full enumeration of the RSet
        self.epsilon = RashomonThreshold      # current tuned epsilon (will be updated)
        self.RashomonThresholdType = RashomonThresholdType
        self.auto_tune_epsilon = auto_tune_epsilon 
        self.static_config = DEFAULT_STATIC_CONFIG.copy()
        self.static_config["regularization"] = self.regularization
        self.tf = None
        self.all_trees = []
        self.trees_in_scope = []
        self.X_train_current = pd.DataFrame() # Stores cumulative DataFrame
        self.y_train_current = pd.Series()      # Stores cumulative Series

        # Attributes for epsilon tuning
        self.predictions_all_trees = None  # Store predictions of all_trees on X_train_current (NumPy array)
        self.accuracy_ordering = None      # Stores indices of all_trees sorted by accuracy (descending)
        
        # Attributes for LFR
        self.epsilon_at_last_full_refit = RashomonThreshold # Stores the tuned epsilon from the last full Rashomon set enumeration
        self.last_full_refit_iteration_count = 0 # Stores the iteration when the last full enumeration happened
        self.current_iteration_from_lp = 0 # Stores the current iteration from LP.py for use within methods


    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Performs a full fit of the TreeFarms model using the maximum epsilon (full enumeration of Rashomon set).
        This also updates the current cumulative training data.
        """

        ## Update current cumulative training data ##
        self.X_train_current = X_train_df.copy()
        self.y_train_current = y_train_series.copy()

        ## Configure TreeFarms for a full fit ##
        config = self.static_config.copy()
        if self.RashomonThresholdType == "Adder":
            config['rashomon_bound_adder'] = self.full_epsilon
        elif self.RashomonThresholdType == "Multiplier":
            config['rashomon_bound_multiplier'] = self.full_epsilon
        else:
            warnings.warn(f"Unsupported RashomonThresholdType: {self.RashomonThresholdType}. Defaulting to 'Adder' logic.")
            config['rashomon_bound_adder'] = self.full_epsilon

        ## Fit TREEFARMS ##
        self.tf = tf(config)
        self.tf.fit(self.X_train_current, self.y_train_current)

        self.all_trees = [self.tf[i] for i in range(self.tf.get_tree_count())]
        self.trees_in_scope = self.all_trees.copy() 

        ### Epsilon Tuning for Full Fit ##
        if not self.all_trees:
            warnings.warn(f"LFRPredictor.fit() completed but found no trees. Cannot tune epsilon.")
            self.epsilon = self.full_epsilon # Default to max epsilon if no trees found
            self.predictions_all_trees = None
            self.accuracy_ordering = None
            
            # Update LFR state even if no trees are found, as this counts as a "full refit"
            self.epsilon_at_last_full_refit = self.epsilon
            self.last_full_refit_iteration_count = self.current_iteration_from_lp
            return
        
        ### Autotune or not ###
        if self.auto_tune_epsilon:
            # Get and sort accuracies descendingly for tuning
            all_accuracies = np.array([_score_single_tree(tree, self.X_train_current.values, self.y_train_current.values) for tree in self.all_trees])
            self.accuracy_ordering = np.argsort(all_accuracies)[::-1] 

            # Get predictions
            predictions_raw_list = [np.array(_predict_single_tree(self.all_trees[idx], self.X_train_current.values)) for idx in self.accuracy_ordering]
            self.predictions_all_trees = np.array(predictions_raw_list).T

            # Tune epsilon and update trees_in_scope based on these accuracies
            self._set_epsilon_and_scope(all_accuracies[self.accuracy_ordering], auto_tune_flag=True) 
        else:
            # If not auto-tuning, set epsilon to full_epsilon and filter scope based on it
            all_accuracies = np.array([_score_single_tree(tree, self.X_train_current.values, self.y_train_current.values) for tree in self.all_trees])
            
            # For fixed epsilon, we still need sorted accuracies for predictions_all_trees for consistency
            self.accuracy_ordering = np.argsort(all_accuracies)[::-1]
            predictions_raw_list = [np.array(_predict_single_tree(self.all_trees[idx], self.X_train_current.values)) for idx in self.accuracy_ordering]
            self.predictions_all_trees = np.array(predictions_raw_list).T
            
            # Call the renamed helper with auto_tune_flag=False
            self._set_epsilon_and_scope(all_accuracies[self.accuracy_ordering], auto_tune_flag=False)
        
        # Update epsilon and iteration
        self.epsilon_at_last_full_refit = self.epsilon                          # Store the tuned epsilon from this full refit
        self.last_full_refit_iteration_count = self.current_iteration_from_lp # Store the iteration when this full refit happened

    ### Refit ###
    def refit(self, X_to_add: pd.DataFrame, y_to_add: pd.Series,
              nominal_rashomon_threshold_input: float, # The fixed epsilon input from SimulationConfigInput
              current_iteration: int, # The iteration from LP.py
              current_train_set_size: int, # len(df_Train) from LP.py
              verbose: bool = False):
        """
        Performs an incremental online update or triggers a full re-enumeration based on Theorem 5.1's condition.
        """

        # Store current iteration count for use in fit method if full refit occurs within this call
        self.current_iteration_from_lp = current_iteration

        ## Concatenate new data with existing cumulative training data ##
        all_X = pd.concat([self.X_train_current, X_to_add], ignore_index=True)
        all_y = pd.concat([self.y_train_current, y_to_add], ignore_index=True)

        ## Update the cumulative training data stored internally ##
        self.X_train_current = all_X
        self.y_train_current = all_y

        perform_full_refit_this_time = False

        # Defensive check to prevent division by zero for very small training sets
        if current_train_set_size == 0:
            if verbose: print("LFR Decision: Full refit due to empty training set (division by zero avoided).")
            perform_full_refit_this_time = True
        else:
            # LHS and RHS #
            # import pdb
            # pdb.set_trace()
            alg1_lhs = self.epsilon_at_last_full_refit - nominal_rashomon_threshold_input
            iteration_difference = max(1, current_iteration - self.last_full_refit_iteration_count)
            alg1_rhs = 2 * (iteration_difference / current_train_set_size)

            # If LHS < RHS, it implies the Rashomon set is likely to change substantially, so perform FULL REFIT.
            if alg1_lhs < alg1_rhs:
                if verbose:
                    print(f"LFR Decision: Full refit due to Theorem 5.1 (Alg 1 condition LHS={alg1_lhs:.4f} < RHS={alg1_rhs:.4f}).")
                perform_full_refit_this_time = True
            else:
                if verbose:
                    print(f"LFR Decision: Online update (subsetting) due to Theorem 5.1 (Alg 1 condition LHS={alg1_lhs:.4f} < RHS={alg1_rhs:.4f}).")

        # Execute the decision: full refit or online update (subsetting)
        if perform_full_refit_this_time:
            self.full_epsilon = 2*nominal_rashomon_threshold_input # Update full_epsilon to the new input value
            self.fit(self.X_train_current, self.y_train_current) # Call self.fit for a full retraining
        else:
            # Online update: recalculate accuracies of all trees on the new cumulative data
            predictions_raw_list = []
            if self.all_trees: # Only attempt to predict if there are trees
                predictions_raw_list = [np.array(_predict_single_tree(tree, self.X_train_current.values)) for tree in self.all_trees]
            
            # This ensures predictions_all_trees has correct shape even if no trees.
            num_samples_current = self.X_train_current.shape[0] # Get current num samples
            if predictions_raw_list:
                self.predictions_all_trees = np.array(predictions_raw_list).T
            else:
                self.predictions_all_trees = np.empty((num_samples_current, 0), dtype=object) 

            objectives = np.array([_score_single_tree(tree, self.X_train_current.values, self.y_train_current.values) for tree in self.all_trees])

            # Handle case where objectives might be empty or all NaNs if no trees are found
            if not objectives.size > 0 or np.all(np.isnan(objectives)):
                warnings.warn("No valid objectives calculated during refit tuning. Skipping ensemble update.")
                self.accuracy_ordering = np.array([])
                self._tune_eps(np.array([])) 
                return 

            # Update accuracy ordering to reflect new data's performance (descending order)
            map_cur_to_new_ordering = np.argsort(objectives)[::-1]
            self.accuracy_ordering = self.accuracy_ordering[map_cur_to_new_ordering]
            self.predictions_all_trees = self.predictions_all_trees[:, map_cur_to_new_ordering]

            # Tune epsilon based on updated accuracies
            self._set_epsilon_and_scope(objectives[self.accuracy_ordering], auto_tune_flag=self.auto_tune_epsilon)

        ### Return Decision ro refit or not ###
        return perform_full_refit_this_time 

    ### Epsilon Tuning Helper ###
    def _set_epsilon_and_scope(self, sorted_accs: np.ndarray, auto_tune_flag: bool):
        """
        Tunes epsilon by finding the ensemble size that maximizes accuracy on training data.
        Updates self.trees_in_scope and self.epsilon.
        Assumes sorted_accs are accuracies of all_trees, sorted DESCENDINGLY (best to worst).
        """

        if not sorted_accs.size > 0:
            warnings.warn("Cannot tune epsilon on an empty set of accuracies. Defaulting to max epsilon.")
            self.trees_in_scope = []
            self.epsilon = self.full_epsilon
            return
        
        # If auto-tuning is enabled
        if auto_tune_flag: 
            
            # Set Up #
            best_num_trees = 0
            best_acc = -1.0

            # For sorted accuracies #
            for i in range(sorted_accs.size):
                if self.predictions_all_trees.shape[1] == 0: # Defensive check if predictions_all_trees somehow empty
                    warnings.warn("predictions_all_trees has no columns during _tune_eps. Cannot compute ensemble accuracy.")
                    y_hat = np.array([]) # Default to empty prediction
                elif i >= self.predictions_all_trees.shape[1]: # Defensive: prevent index out of bounds
                    # If i+1 is beyond actual number of trees, use all available
                    current_ensemble_preds = self.predictions_all_trees
                else:
                    current_ensemble_preds = self.predictions_all_trees[:, :i+1]

                if current_ensemble_preds.size == 0:
                    acc = -1.0 # Cannot compute accuracy
                else:
                    if len(np.unique(self.y_train_current)) > 2: # Multi-class
                        y_hat_val, _ = stats.mode(current_ensemble_preds, axis=1, keepdims=False)
                        y_hat = y_hat_val.squeeze()
                    else: # Binary classification (0/1)
                        y_hat = (current_ensemble_preds.mean(axis=1) > 0.5).astype(int)

                    if self.y_train_current.empty: # Defensive: avoid comparing to empty y_train_current
                        acc = -1.0
                    else:
                        acc = np.mean(y_hat == self.y_train_current.values)

                if acc > best_acc:
                    best_acc = acc
                    best_num_trees = i + 1

            if best_num_trees > 0:
                self.trees_in_scope = [self.all_trees[k] for k in self.accuracy_ordering[:best_num_trees]]
            else:
                warnings.warn("Epsilon tuning resulted in 0 optimal trees. Retaining all trees from full fit.")
                self.trees_in_scope = self.all_trees.copy()
                best_num_trees = len(self.all_trees)

            if best_num_trees == len(self.all_trees):
                self.epsilon = self.full_epsilon
            else:
                self.epsilon = sorted_accs[0] - sorted_accs[best_num_trees - 1]
                self.epsilon = max(0.0, min(self.epsilon, self.full_epsilon))
        else: # Fixed Epsilon
            self.epsilon = self.full_epsilon # Epsilon is fixed to the full_epsilon/RashomonThreshold input

            # Set trees_in_scope based on this fixed epsilon
            if sorted_accs.size == 0:
                self.trees_in_scope = []
                return

            best_accuracy_overall = sorted_accs[0] # Best accuracy among all_trees
            errors_relative_to_best = best_accuracy_overall - sorted_accs
            
            trees_meeting_epsilon = [self.all_trees[k] for i, k in enumerate(self.accuracy_ordering) 
                                     if errors_relative_to_best[i] <= self.epsilon]
            
            if not trees_meeting_epsilon:
                warnings.warn(f"Fixed epsilon ({self.epsilon}) resulted in 0 trees in scope. Retaining all trees from full fit.")
                self.trees_in_scope = self.all_trees.copy()
            else:
                self.trees_in_scope = self.all_trees.copy()#trees_meeting_epsilon


    ### Helper to get predictions from all trees ###
    def _get_ensemble_predictions_df(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        if not self.trees_in_scope:
            warnings.warn("No trees currently in scope for ensemble predictions. Returning empty DataFrame.")
            return pd.DataFrame(index=X_data_df.index, columns=[])

        X_data_np = X_data_df.values 

        # Get predictions from each individual tree.
        predictions_list_of_lists = [_predict_single_tree(tree, X_data_np) for tree in self.trees_in_scope]
        ensemble_predictions_df = pd.DataFrame(np.array(predictions_list_of_lists).T) 
        ensemble_predictions_df.columns = [f"Tree_{i}" for i in range(ensemble_predictions_df.shape[1])]
        ensemble_predictions_df.index = X_data_df.index

        return ensemble_predictions_df
    
    ### Predict Model (Ensemble Mode Prediction) ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        if self.tf is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not self.trees_in_scope:
             warnings.warn("No trees in scope for prediction. Returning zeros.")
             return np.zeros(X_data_df.shape[0], dtype=int)
        ensemble_predictions_df = self._get_ensemble_predictions_df(X_data_df)
        mode_predictions = stats.mode(ensemble_predictions_df, axis=1)[0].squeeze()
        return mode_predictions.astype(int)

    ### Get prediction probabilities ###
    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int = None) -> torch.Tensor:
        if self.tf is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not self.trees_in_scope:
            num_samples = X_data_np.shape[0]
            num_classes = len(np.unique(self.y_train_current)) if self.y_train_current is not None and not self.y_train_current.empty else 2
            warnings.warn("No trees in scope for predict_proba_K. Returning default tensor.")
            return torch.full((num_samples, 0, num_classes), -float('inf'), dtype=torch.float32)

        X_data_df = pd.DataFrame(X_data_np, columns=self.X_train_current.columns)
        ensemble_predictions_df = self._get_ensemble_predictions_df(X_data_df)

        # ### DEBUG Save ###
        # import os
        # filename = "/Users/simondn/Downloads/AutoTune1/"
        # base_name, ext = os.path.splitext(filename)
        # counter = 0
        # new_filename = filename

        # while os.path.exists(new_filename):
        #     counter += 1
        #     new_filename = f"{base_name}_{counter}{ext}"

        # ensemble_predictions_df.to_csv(new_filename, index=False)
        # print(f"File saved as: {new_filename}")
        # ######

        num_samples = ensemble_predictions_df.shape[0]
        num_trees_in_ensemble = ensemble_predictions_df.shape[1]

        unique_classes_sorted = np.sort(np.unique(self.y_train_current))
        num_classes = len(unique_classes_sorted)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes_sorted)}

        log_probs_N_K_C = torch.full((num_samples, num_trees_in_ensemble, num_classes), -float('inf'), dtype=torch.float32)

        for sample_idx in range(num_samples):
            for tree_idx in range(num_trees_in_ensemble):
                predicted_class_label = ensemble_predictions_df.iloc[sample_idx, tree_idx]
                mapped_class_index = class_mapping.get(predicted_class_label)
                if mapped_class_index is not None:
                    log_probs_N_K_C[sample_idx, tree_idx, mapped_class_index] = 0.0

        return log_probs_N_K_C

    ### Get total and unique tree counts ###
    def get_tree_counts(self) -> dict:
        if self.tf is None:
            return {"AllTreeCount": 0, "UniqueTreeCount": 0}

        all_tree_count = self.tf.get_tree_count()
        unique_tree_count = len(self.trees_in_scope)

        return {"AllTreeCount": all_tree_count, "UniqueTreeCount": unique_tree_count}

    ### Get predictions from each individual tree ###
    def get_raw_ensemble_predictions(self, X_data_df: pd.DataFrame) -> pd.DataFrame:
        if self.tf is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not self.trees_in_scope:
            warnings.warn("No trees found in the Rashomon set for raw ensemble predictions. Returning empty DataFrame.")
            return pd.DataFrame(index=X_data_df.index, columns=[])
        
        return self._get_ensemble_predictions_df(X_data_df)