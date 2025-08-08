# Summary: Chooses an index at random from the candidate set to be queried.

### Libraries ###
import pandas as pd
import numpy as np

### Passive Learning ###
class PassiveLearningSelector:
    def __init__(self, BatchSize: int = 1, Seed: int = None, **kwargs):
        self.BatchSize = BatchSize
        self.Seed = Seed

    def select(self, df_Candidate: pd.DataFrame, **kwargs) -> dict:
        if df_Candidate.shape[0] >= self.BatchSize:
             QueryObservation = df_Candidate.sample(n=self.BatchSize, random_state=self.Seed) 
             IndexRecommendation = list(QueryObservation.index)
        else:
            IndexRecommendation = list(df_Candidate.index) 

        return {"IndexRecommendation": IndexRecommendation}