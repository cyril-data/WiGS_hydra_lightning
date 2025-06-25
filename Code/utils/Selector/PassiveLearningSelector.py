# Summary: Chooses an index at random from the candidate set to be queried.

### Libraries ###
import pandas as pd
import numpy as np

class PassiveLearningSelector:
    def __init__(self, BatchSize: int = 5, Seed: int = None, **kwargs):
        self.BatchSize = BatchSize
        self.Seed = Seed

    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, auxiliary_columns: list = None) -> dict:
        if df_Candidate.shape[0] >= self.BatchSize:
             QueryObservation = df_Candidate.sample(n=self.BatchSize, random_state=self.Seed) 
             IndexRecommendation = list(QueryObservation.index)
        else:
            IndexRecommendation = list(df_Candidate.index) # Select all if fewer than BatchSize

        return {"IndexRecommendation": IndexRecommendation}