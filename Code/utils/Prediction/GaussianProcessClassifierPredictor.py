# Summary: Implements a Gaussian Process Classifier predictor.

### Libraries ###
import pandas as pd
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, WhiteKernel # Common kernels
from sklearn.preprocessing import LabelEncoder # For handling non-numeric target labels

class GaussianProcessClassifierPredictor:
    def __init__(self, Seed: int,
                 kernel_type: str = 'RBF', # e.g., 'RBF', 'DotProduct', 'Matern', 'ConstantKernel'
                 kernel_length_scale: float = 1.0,
                 kernel_nu: float = 1.5, # For Matern kernel
                 optimizer: str = 'fmin_l_bfgs_b', # Optimizer for kernel hyperparameters
                 n_restarts_optimizer: int = 0, # Number of restarts for the optimizer
                 max_iter_predict: int = 100 # Max iterations for predict_proba (for approximations)
                 , **kwargs):
        """
        Initializes the GaussianProcessClassifierPredictor.

        Args:
            Seed (int): Seed for reproducibility.
            kernel_type (str, optional): Type of kernel. Defaults to 'RBF'.
            kernel_length_scale (float, optional): Length scale for RBF or Matern kernels. Defaults to 1.0.
            kernel_nu (float, optional): Nu parameter for Matern kernel. Defaults to 1.5.
            optimizer (str, optional): Optimizer to use for kernel hyperparameters. Defaults to 'fmin_l_bfgs_b'.
            n_restarts_optimizer (int, optional): Number of restarts for the optimizer. Defaults to 0.
            max_iter_predict (int, optional): Maximum number of iterations for predict_proba (for approximations like EP).
            **kwargs: Catches any other arguments not used by this model's init.
        """
        self.Seed = Seed
        self.kernel_type = kernel_type
        self.kernel_length_scale = kernel_length_scale
        self.kernel_nu = kernel_nu
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict

        self.model = None
        self.label_encoder = LabelEncoder() # To handle non-numeric target labels
        self.unique_classes = None # To store unique classes from training data

        np.random.seed(self.Seed) # Set numpy seed for reproducibility

    def _get_kernel(self):
        """Helper to construct the kernel based on type and parameters."""
        if self.kernel_type == 'RBF':
            return RBF(length_scale=self.kernel_length_scale)
        elif self.kernel_type == 'DotProduct':
            return DotProduct(sigma_0=self.kernel_length_scale) # length_scale re-used for sigma_0
        elif self.kernel_type == 'Matern':
            return Matern(length_scale=self.kernel_length_scale, nu=self.kernel_nu)
        elif self.kernel_type == 'ConstantKernel':
            return ConstantKernel(constant_value=self.kernel_length_scale) # length_scale re-used for constant_value
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Fits the Gaussian Process Classifier model.

        Args:
            X_train_df (pd.DataFrame): The training features.
            y_train_series (pd.Series): The training target.
        """
        # Convert target labels to numeric using LabelEncoder
        y_train_encoded = self.label_encoder.fit_transform(y_train_series)
        self.unique_classes = self.label_encoder.classes_ # Store original class labels

        # Initialize GPC with the selected kernel and optimizer
        kernel = self._get_kernel()
        self.model = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.Seed, # For reproducibility of internal random processes
            max_iter_predict=self.max_iter_predict # For the predict_proba approximation
        )
        self.model.fit(X_train_df.values, y_train_encoded) # Fit with NumPy array

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions (hard class labels) on new data.

        Args:
            X_data_df (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # Predict encoded labels and then inverse transform to original labels
        predicted_encoded_labels = self.model.predict(X_data_df.values)
        return self.label_encoder.inverse_transform(predicted_encoded_labels)

    def predict_proba(self, X_data_df: pd.DataFrame) -> np.ndarray:
        """
        Makes probability predictions on new data.

        Args:
            X_data_df (pd.DataFrame): The data to make probability predictions on.

        Returns:
            np.ndarray: Predicted class probabilities (shape: n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        return self.model.predict_proba(X_data_df.values)

    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int) -> torch.Tensor:
        """
        Generates K 'samples' of log-probabilities.
        For sklearn.GPC, this involves duplicating the deterministic predict_proba output K times.
        (Does NOT represent true Bayesian posterior sampling of latent functions).

        Args:
            X_data_np (np.ndarray): NumPy array of features for prediction.
            K_samples (int): Number of 'samples' to generate (number of duplications).

        Returns:
            torch.Tensor: Log probabilities of shape (N_samples, K_samples, num_classes).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if self.unique_classes is None:
             raise RuntimeError("Model has not been fitted, unique_classes not determined.")

        # Get the single, deterministic probability prediction
        probs_N_C = self.model.predict_proba(X_data_np)

        # Convert to log probabilities, add epsilon to avoid log(0)
        log_probs_N_C = np.log(probs_N_C + np.finfo(float).eps)

        # Convert to PyTorch tensor
        log_probs_tensor = torch.tensor(log_probs_N_C, dtype=torch.float32)

        # Duplicate the log_probs_N_C tensor K_samples times along a new dimension (dim=1)
        # This creates a tensor of shape (N_samples, K_samples, num_classes)
        log_probs_N_K_C = log_probs_tensor.unsqueeze(1).repeat(1, K_samples, 1)

        return log_probs_N_K_C

    # GPC does not have 'trees' or ensemble counts in the same way as TreeFarms or RandomForest.
    # So, get_tree_counts and get_raw_ensemble_predictions methods are not implemented for GPC.
    # Models that use predict_proba_K but don't have get_tree_counts will fall through
    # LearningProcedure's hasattr(predictor_model, 'get_tree_counts') check.