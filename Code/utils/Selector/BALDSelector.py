### Libraries ###
import math
import torch
import numpy as np
import pandas as pd 
from typing import List
from dataclasses import dataclass
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Toma ###
class TomaExecute:
    def chunked(self, data, chunk_size):
        def decorator(func):
            def wrapper(*args, **kwargs):
                N = data.shape[0]
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    func(data[start:end], start, end)
            return wrapper
        return decorator
class Toma:
    def __init__(self):
        self.execute = TomaExecute()
toma = Toma()

### Conditional Entropy Function ###
def ComputeConditionalEntropyFunction(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        EntropyVals = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        entropies_N[start:end].copy_(-torch.sum(EntropyVals, dim=(1, 2)) / K)
    
    return entropies_N

### Marginal Entropy Function ###
def ComputeEntropyFunction(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
    
    return entropies_N

### Data Class for Output ###
@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


class BALDSelector:

    ### Initialize ###
    def __init__(self, BatchSize: int, K_BALD_Samples: int = 20, Seed: int = None, **kwargs):
        self.BatchSize = BatchSize
        self.K_BALD_Samples = K_BALD_Samples
        self.Seed = Seed 

    ### Selection ###
    def select(self, df_Candidate: pd.DataFrame, Model, df_Train: pd.DataFrame = None, auxiliary_columns: list = None) -> dict:
        if not hasattr(Model, 'predict_proba_K'):
            raise AttributeError("BALDSelector requires the provided Model to have a 'predict_proba_K' method.")

        ## Set up df_Candidate features ##
        X_candidate_df, _ = get_features_and_target(
            df=df_Candidate,
            target_column_name="Y", 
            auxiliary_columns=auxiliary_columns)
        X_candidate_np = X_candidate_df.values

        ## Generate log_probs_N_K_C using the model's prediction method ##
        log_probs_N_K_C = Model.predict_proba_K(X_candidate_np, self.K_BALD_Samples)

        #$ Determine dimensions ##
        N_candidate = X_candidate_np.shape[0]
        batch_size_actual = min(self.BatchSize, N_candidate)

        ## Compute BALD scores ##
        EnsembleEntropy = ComputeEntropyFunction(log_probs_N_K_C)
        ConditionalEntropy = ComputeConditionalEntropyFunction(log_probs_N_K_C)
        UncertaintyMetrics = EnsembleEntropy - ConditionalEntropy

        ## Get the top observations ##
        top_scores, top_local_indices = torch.topk(UncertaintyMetrics, batch_size_actual, largest=True, sorted=False)

        ## Convert the local indices ##
        candidate_df_indices = df_Candidate.index.values 
        IndexRecommendation = candidate_df_indices[top_local_indices.cpu().numpy().astype(int)].tolist()

        return {"IndexRecommendation": IndexRecommendation}