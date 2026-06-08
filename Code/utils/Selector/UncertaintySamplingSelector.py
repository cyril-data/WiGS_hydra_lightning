### Library ###
import numpy as np
import pandas as pd
from utils.Auxiliary.DataFrameUtils import get_features_and_target


### Function ###
class UncertaintySamplingSelector:
    """
    Selects the candidate point x that maximizes the model's predictive uncertainty.

    This selector is MODEL AGNOSTIC. It relies on the model wrapper having a
    method `predict_with_std(X)` that returns (mean, standard_deviation).
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Ignored arguments.
        """
        pass

    def select(
        self, df_Candidate: pd.DataFrame, df_Train: pd.DataFrame, y_size: int, Model=None, **kwargs
    ) -> dict:
        """
        Selects the candidate with the highest predictive standard deviation.
        """
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        # 1. Get Candidate Features
        X_cand, _ = get_features_and_target(df_Candidate, y_size)

        # 2. Ask the Model for Uncertainty
        if not hasattr(Model, "predict_with_std"):
            raise AttributeError(
                f"The model {type(Model).__name__} does not implement 'predict_with_std'."
            )

        _, uncertainties = Model.predict_with_std(X_cand)

        # 3. Handle Edge Cases (e.g. if uncertainties is 1D or 2D)
        if uncertainties.ndim > 1:
            uncertainties = uncertainties.flatten()

        # 4. Select Max Uncertainty
        best_loc = np.argmax(uncertainties)
        best_index = df_Candidate.iloc[[best_loc]].index[0]

        return {"IndexRecommendation": [float(best_index)]}
