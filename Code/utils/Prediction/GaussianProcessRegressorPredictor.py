### Libraries ###
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class GaussianProcessRegressorPredictor:
    """
    A wrapper for the scikit-learn GaussianProcessRegressor model.

    Uses a composite kernel: ConstantKernel * RBF + WhiteKernel.
    - ConstantKernel * RBF captures the signal (amplitude and lengthscale).
    - WhiteKernel captures observation noise (aleatoric uncertainty).

    The kernel hyperparameters are optimized via marginal likelihood
    maximization during each call to .fit().
    """

    ### Initialize Model ###
    def __init__(self, alpha: float = 1e-7, n_restarts_optimizer: int = 3, **kwargs):
        """
        Args:
            alpha (float): Value added to the diagonal of the kernel matrix during
                fitting for numerical stability. This is distinct from the kernel's
                WhiteKernel, which learns the noise level.
            n_restarts_optimizer (int): Number of restarts of the optimizer for
                finding the kernel hyperparameters that maximize the log-marginal
                likelihood. More restarts reduce the risk of local optima but
                increase fitting time.
            **kwargs: Accepts and ignores additional keyword arguments for
                consistency with the predictor interface.
        """
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = None

    ### Fit Model ###
    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        """
        Fits a new GaussianProcessRegressor on the provided training data.

        A fresh kernel is constructed at each call to avoid carrying over
        stale hyperparameter values from a previous fit, which could bias
        the optimizer toward a suboptimal local minimum as the training
        set evolves during active learning.
        """
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-1, (1e-5, 1e1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=42
        )
        self.model.fit(X_train_df, y_train_series)

    ### Predict Model ###
    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        """
        Returns point predictions (posterior mean) for the given features.
        """
        return self.model.predict(X_data_df)

    ### Predict with Uncertainty ###
    def predict_with_std(self, X_data_df: pd.DataFrame):
        """
        Returns both the posterior mean and standard deviation.

        This is not required by the base predictor interface, but is
        available for future use (e.g., a GP-specific uncertainty
        sampling selector).

        Returns:
            tuple: (y_mean, y_std) where both are np.ndarrays.
        """
        return self.model.predict(X_data_df, return_std=True)