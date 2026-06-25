### Libraries ###
import pandas as pd

# import numpy as np


### Passive Learning ###
class PassiveLearningSelector:
    """
    Implements a random sampling selection strategy.

    Attributes:
        Seed (int): A random seed to ensure the sampling is reproducible.
    """

    def __init__(self, Seed: int = None, k_top_candidate=1, **kwargs):
        """
        Initializes the PassiveLearningSelector.

        Args:
            Seed (int, optional): The random seed for reproducibility of the sampling process.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """
        self.Seed = Seed
        self.k_top_candidate = k_top_candidate

    def select(self, df_Candidate: pd.DataFrame, **kwargs) -> dict:
        """
        Randomly selects a single observation from the candidate set.

        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points from which to select.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.

        Returns:
            dict: A dictionary containing the recommended point's index, in the format `{'IndexRecommendation': [index]}`.
        """

        QueryObservation = df_Candidate.sample(
            n=min(self.k_top_candidate, len(df_Candidate)), random_state=self.Seed
        )
        IndexRecommendation = list(QueryObservation.index)

        return {"IndexRecommendation": IndexRecommendation}
