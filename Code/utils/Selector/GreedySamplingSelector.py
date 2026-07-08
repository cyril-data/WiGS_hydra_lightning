### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target
import time

from utils.Prediction.LightHydra import (
    hl_pd_to_dataloader,
    hl_y_pred_pd_to_tensor,
    hl_np_to_dataloader,
)
import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def min_dist_per_row_gpu(candidates_np, ref_np, batch_size, train_chunk_size=None):
    """
    Pour chaque ligne de `candidates_np`, calcule la distance euclidienne minimale
    par rapport à toutes les lignes de `ref_np`, en GPU et par batchs (memory-efficient).
    Utilisé pour GSx (features) et GSy (target/predictions).
    """
    if train_chunk_size is None:
        train_chunk_size = batch_size

    ref_t = torch.from_numpy(ref_np).to(DEVICE).float()
    ref_sq = (ref_t**2).sum(dim=1)
    n_ref = ref_t.shape[0]

    n = len(candidates_np)
    min_dist = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = (
                torch.from_numpy(candidates_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .float()
            )
            b_sq = (b**2).sum(dim=1, keepdim=True)

            running_min = torch.full(
                (b.shape[0],), float("inf"), device=DEVICE, dtype=torch.float16
            )

            for j in range(0, n_ref, train_chunk_size):
                ref_chunk = ref_t[j : j + train_chunk_size]
                ref_sq_chunk = ref_sq[j : j + train_chunk_size]

                dist_sq = b_sq + ref_sq_chunk.unsqueeze(0) - 2.0 * (b @ ref_chunk.T)
                dist_sq.clamp_(min=0)
                dist = dist_sq.sqrt_()

                running_min = torch.minimum(running_min, dist.min(dim=1).values)

                del dist_sq, dist

            min_dist[i : i + batch_size] = running_min.float().cpu().numpy()

            del b, b_sq, running_min

    return min_dist


def min_igs_per_row_gpu(
    X_candidate_np,
    X_train_np,
    Y_candidate_np,
    Y_train_np,
    batch_size,
    train_chunk_size=None,
):
    """
    Pour chaque ligne candidate, calcule min_m( d_X(n,m) * d_Y(n,m) ) sur les points
    d'entraînement m, en GPU et par batchs. Utilisé pour la stratégie iGS.
    """
    if train_chunk_size is None:
        train_chunk_size = batch_size

    X_train_t = torch.from_numpy(X_train_np).to(DEVICE).float()
    X_train_sq = (X_train_t**2).sum(dim=1)
    n_train_x = X_train_t.shape[0]

    Y_train_t = torch.from_numpy(Y_train_np).to(DEVICE).float()
    Y_train_sq = (Y_train_t**2).sum(dim=1)
    n_train_y = Y_train_t.shape[0]

    assert (
        n_train_x == n_train_y
    ), "X_train et Y_train doivent avoir le même nombre de lignes (même index)"

    n = len(X_candidate_np)
    min_dxy = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            bX = (
                torch.from_numpy(X_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .float()
            )
            bX_sq = (bX**2).sum(dim=1, keepdim=True)

            bY = (
                torch.from_numpy(Y_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .float()
            )
            bY_sq = (bY**2).sum(dim=1, keepdim=True)

            running_min = torch.full(
                (bX.shape[0],), float("inf"), device=DEVICE, dtype=torch.float16
            )

            for j in range(0, n_train_x, train_chunk_size):
                Xt_chunk = X_train_t[j : j + train_chunk_size]
                Xsq_chunk = X_train_sq[j : j + train_chunk_size]
                Yt_chunk = Y_train_t[j : j + train_chunk_size]
                Ysq_chunk = Y_train_sq[j : j + train_chunk_size]

                dX_chunk = (
                    (bX_sq + Xsq_chunk.unsqueeze(0) - 2.0 * (bX @ Xt_chunk.T))
                    .clamp_(min=0)
                    .sqrt_()
                )
                dY_chunk = (
                    (bY_sq + Ysq_chunk.unsqueeze(0) - 2.0 * (bY @ Yt_chunk.T))
                    .clamp_(min=0)
                    .sqrt_()
                )

                dxy_chunk = dX_chunk * dY_chunk

                running_min = torch.minimum(running_min, dxy_chunk.min(dim=1).values)

                del dX_chunk, dY_chunk, dxy_chunk

            min_dxy[i : i + batch_size] = running_min.float().cpu().numpy()

            del bX, bY, bX_sq, bY_sq, running_min

    return min_dxy


class GreedySamplingSelector:
    """
    Implements the greedy sampling methods from Wu, Lin, and Huang (2018).

    Attributes:
        strategy (str): The active strategy to be used ('GSx', 'GSy', or 'iGS').
        distance (str): The distance metric used for calculations (e.g., 'euclidean').
        Seed (int): The random seed for reproducibility (Note: not used in current
            implementation but retained for API consistency).
    """

    ### Initialize ###
    def __init__(
        self,
        strategy: str,
        distance: str = "euclidean",
        Seed: int = None,
        k_top_candidate=1,
        **kwargs,
    ):
        """
        Initializes the GreedySamplingSelector.

        Args:
            strategy (str): The greedy sampling strategy. Must be one of 'GSx', 'GSy', or 'iGS'.
            distance (str, optional): The distance metric to use, compatible with `scipy.spatial.distance.cdist`.
            Seed (int, optional): A random seed for reproducibility.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """
        if strategy not in ["GSx", "GSy", "iGS"]:
            raise ValueError(
                f"Invalid greedy sampling strategy: {strategy}. Must be 'GSx', 'GSy', or 'iGS'."
            )
        self.strategy = strategy
        self.distance = distance
        self.Seed = Seed
        self.k_top_candidate = k_top_candidate

    ### Select Observation(s) ###
    def select(
        self,
        df_Candidate: pd.DataFrame,
        y_size: int,
        Model=None,
        df_Train: pd.DataFrame = None,
        SimulationConfigInputUpdated: dict = None,
        **kwargs,
    ) -> dict:
        """
        Selects the top-k most informative observations from the candidate set.
        Args:
            df_Candidate (pd.DataFrame): The pool of unlabeled data points from which to select.
            y_size (int): Number of target columns.
            Model (object, optional): A trained model.
            df_Train (pd.DataFrame, optional): The current set of labeled training data.
            SimulationConfigInputUpdated (dict, optional): Simulation config; if it contains a
                non-None "hl_trainer", predictions are computed through the LightHydra pipeline
                instead of a plain `Model.predict`.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        Returns:
            dict: A dictionary containing the recommended points' indices, in the format
                `{'IndexRecommendation': [index, ...]}`.
        """

        ## 1. Initialization
        StartTime = time.time()

        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Set up candidate / training features ##
        X_Candidate, _ = get_features_and_target(df_Candidate, y_size=None)
        X_Train, y_Train = get_features_and_target(df_Train, y_size=y_size)

        X_Candidate_f32 = X_Candidate.values.astype(np.float32)
        X_Train_f32 = X_Train.values.astype(np.float32)

        select_ytrain_cols = None

        batch_size = 512

        # print(f"\t+++ GreedySampling #1 : {time.time() - StartTime} +++")

        ## 2. Prediction on candidates (only needed for GSy / iGS) ##
        StartTime = time.time()

        Predictions = None
        if self.strategy in ["GSy", "iGS"]:
            if (
                SimulationConfigInputUpdated is not None
                and SimulationConfigInputUpdated.get("hl_trainer") is not None
            ):
                hl_data = SimulationConfigInputUpdated["hl_data"]

                candidate_labels = X_Candidate.index.tolist()
                hl_data.pred_data.update_indices(candidate_labels)

                y_pred = Model.predict(model=Model.model, dataloaders=hl_data)

                y_pred, select_ytrain_cols = hl_y_pred_pd_to_tensor(
                    y_pred, y_Train.columns.to_list(), X_Candidate.index
                )
                # restrain only on regression
                Predictions = y_pred[select_ytrain_cols].values
            else:
                Predictions = Model.predict(X_Candidate)

        # print(f"\t+++ GreedySampling Prediction on candidates : {time.time() - StartTime} +++")

        if select_ytrain_cols is not None:
            y_Train = y_Train[select_ytrain_cols]

        pred_vals = None
        y_train_values = None
        if Predictions is not None:
            pred_vals = (
                Predictions.reshape(-1, 1).astype(np.float32)
                if len(Predictions.shape) == 1
                else Predictions.astype(np.float32)
            )
            y_train_values = (
                y_Train.values.reshape(-1, 1).astype(np.float32)
                if len(y_Train.shape) == 1
                else y_Train.values.astype(np.float32)
            )

        ## 3. Distance / score calculation (GPU, batched) ##
        StartTime = time.time()

        final_scores = None

        if self.strategy == "GSx":
            final_scores = min_dist_per_row_gpu(X_Candidate_f32, X_Train_f32, batch_size)

        elif self.strategy == "GSy":
            final_scores = min_dist_per_row_gpu(pred_vals, y_train_values, batch_size)

        elif self.strategy == "iGS":
            final_scores = min_igs_per_row_gpu(
                X_Candidate_f32, X_Train_f32, pred_vals, y_train_values, batch_size
            )

        # print(f"\t+++ GreedySampling final_scores : {time.time() - StartTime} +++")

        ## 4. Top-k selection (largest scores = most informative) ##
        top_k_number = self.k_top_candidate
        if len(final_scores) < self.k_top_candidate:
            top_k_number = len(final_scores)

        best_candidate_iloc = np.argpartition(final_scores, -top_k_number)[-top_k_number:]

        ## Output ##
        IndexRecommendation = df_Candidate.iloc[best_candidate_iloc].index.to_list()

        return {"IndexRecommendation": IndexRecommendation}

    # ### Select Observation ###
    # def select_np(
    #     self,
    #     df_Candidate: pd.DataFrame,
    #     y_size: int,
    #     Model=None,
    #     df_Train: pd.DataFrame = None,
    #     **kwargs,
    # ) -> dict:
    #     """
    #     Selects the single most informative observation from the candidate set.

    #     Args:
    #         df_Candidate (pd.DataFrame): The pool of unlabeled data points from which to select.
    #         Model (object, optional): A trained model.
    #         df_Train (pd.DataFrame, optional): The current set of labeled training data.
    #         **kwargs: Accepts and ignores additional keyword arguments for consistency.

    #     Returns:
    #         dict: A dictionary containing the recommended point's index, in the format `{'IndexRecommendation': [index]}`.
    #     """

    #     if df_Candidate.empty:
    #         return {"IndexRecommendation": []}

    #     ## Set up candidate features ##

    #     X_Candidate, _ = get_features_and_target(df_Candidate, y_size=None)

    #     X_Candidate_np = X_Candidate.values

    #     ## Set up training features ##
    #     X_Train, y_Train = get_features_and_target(df_Train, y_size=y_size)

    #     X_Train_np = X_Train.values

    #     ## GSx Logic ##
    #     d_nX = None
    #     if self.strategy in ["GSx", "iGS"]:
    #         d_nmX = cdist(X_Candidate_np, X_Train_np, metric=self.distance)
    #         d_nX = d_nmX.min(axis=1)

    #     ## GSy Logic ##
    #     d_nY = None
    #     if self.strategy in ["GSy", "iGS"]:
    #         Predictions = Model.predict(X_Candidate)
    #         d_nmY = cdist(
    #             Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric=self.distance
    #         )
    #         d_nY = d_nmY.min(axis=1)

    #     ## Selection ##
    #     MaxRowNumber = -1
    #     if self.strategy == "GSx":
    #         MaxRowNumber = np.argmax(d_nX)
    #     elif self.strategy == "GSy":
    #         MaxRowNumber = np.argmax(d_nY)
    #     elif self.strategy == "iGS":
    #         if d_nmX is None or d_nmY is None:
    #             raise RuntimeError("iGS strategy requires both GSx and GSy components.")
    #         d_nXY_matrix = d_nmX * d_nmY
    #         d_nXY = d_nXY_matrix.min(axis=1)
    #         print("final_scores nympy", d_nXY)

    #         MaxRowNumber = np.argmax(d_nXY)

    #     ## Output ##
    #     IndexRecommendation = df_Candidate.iloc[[MaxRowNumber]].index[0]
    #     return {"IndexRecommendation": [float(IndexRecommendation)]}
