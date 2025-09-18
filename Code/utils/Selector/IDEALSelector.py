# Summary: Implements IDEAL (Inverse-Distance based Exploration for Active Learning) for regression.
#          Acquisition a(x) = s^2(x) + delta * z(x), where s^2 is IDW variance over residuals and z is an IDW exploration term.

### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target 

class IDEALSelector:
    """
    Pool-based, model-driven AL for regression using inverse-distance weighting (IDW).

    Acquisition (univariate y):
        a(x) = s^2(x) + delta * z(x)
      where:
        s^2(x) = sum_k v_k(x) * (y_k - y_hat(x))^2
        w_k(x) = exp(-d^2(x,x_k)) / d^2(x,x_k)
        v_k(x) = w_k / sum_j w_j      (normalized)
        z(x)   = (2/pi) * arctan( 1 / sum_k w_k(x) ), with z(x)=0 for any already-labeled x
      We choose the candidate(s) with MAX a(x).
    """

    ### Initialize ###
    def __init__(self, 
                 delta: float = 5.0, 
                 BatchSize: int = 1, 
                 Seed: int = None, 
                 eps: float = 1e-12,
                 **kwargs):
        self.delta = float(delta)
        self.BatchSize = int(BatchSize)
        self.Seed = Seed
        self.eps = float(eps)

    ### Internal: [-1,1] min–max scaling per paper (sigma) ###
    def _sigma_scale(self, X_all: np.ndarray, X: np.ndarray) -> np.ndarray:
        xmin = X_all.min(axis=0, keepdims=True)
        xmax = X_all.max(axis=0, keepdims=True)
        # Avoid zero-width features
        span = np.maximum(xmax - xmin, self.eps)
        # sigma_i(x) = 2/(xmax-xmin) * (x - (xmax+xmin)/2) in [-1,1]
        center = (xmax + xmin) / 2.0
        return 2.0 / span * (X - center)

    ### Select Observation ###
    def select(self, 
               df_Candidate: pd.DataFrame, 
               Model=None,
               df_Train: pd.DataFrame = None, 
               auxiliary_columns: list = None,
               **kwargs) -> dict:
        # Basic checks
        if df_Candidate is None or df_Candidate.empty or df_Train is None or df_Train.empty:
            return {"IndexRecommendation": []}
        if Model is None:
            raise ValueError("IDEALSelector requires a trained Model for predictions.")

        ## Set up features/labels ##
        X_Cand, _   = get_features_and_target(df=df_Candidate, target_column_name="Y")
        X_Train, yT = get_features_and_target(df=df_Train, target_column_name="Y")

        Xc = X_Cand.values
        Xt = X_Train.values
        yt = yT.values.reshape(-1)

        ## Paper scaling (scaled Euclidean via sigma) across {candidates ∪ train} ##
        X_all = np.vstack([Xc, Xt])
        Xc_s = self._sigma_scale(X_all, Xc)
        Xt_s = self._sigma_scale(X_all, Xt)

        ## Squared distances d^2(x, x_k) on scaled space ##
        D2 = cdist(Xc_s, Xt_s, metric="sqeuclidean")  # shape: [Nc, Nt]

        ## IDW weights: w_k(x) = exp(-d^2)/d^2 (stabilized) ##
        W = np.exp(-D2) / np.maximum(D2, self.eps)
        Wsum = W.sum(axis=1, keepdims=True)  # [Nc, 1]

        ## Normalized weights v_k(x) ##
        V = W / np.maximum(Wsum, self.eps)

        ## Model predictions on candidates ##
        yhat = np.asarray(Model.predict(X_Cand)).reshape(-1)  # [Nc]

        ## s^2(x): weighted squared residuals to all training labels ##
        resid2 = (yt[None, :] - yhat[:, None]) ** 2  # [Nc, Nt]
        s2 = (V * resid2).sum(axis=1)                # [Nc]

        ## z(x): (2/pi) * arctan(1 / sum_k w_k(x)) with exact zero at duplicate points ##
        z = (2.0 / np.pi) * np.arctan(1.0 / np.maximum(Wsum.reshape(-1), self.eps))  # [Nc]
        # Force z=0 wherever a candidate coincides with any training point (d^2=0)
        has_duplicate = (D2.min(axis=1) <= self.eps)
        z[has_duplicate] = 0.0

        ## Acquisition and selection ##
        a = s2 + self.delta * z
        k = int(max(1, self.BatchSize))
        top_idx_local = np.argsort(-a)[:k]  # descending

        ## Output (follow GreedySamplingSelector style) ##
        rec_indices = df_Candidate.iloc[top_idx_local].index.astype(float).tolist()
        return {"IndexRecommendation": rec_indices}