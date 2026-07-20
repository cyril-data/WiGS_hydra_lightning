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


def min_dist_per_row_gpu(candidates_np, ref_np, batch_size, train_chunk_size=None, dtype=torch.float16):
    """
    Pour chaque ligne de `candidates_np`, calcule la distance euclidienne minimale
    par rapport à toutes les lignes de `ref_np`, en GPU et par batchs (memory-efficient).
    Utilisé pour GSx (features) et GSy (target/predictions).

    dtype controls the precision of the bulk matmul only (the expensive part,
    where fp16 gets much higher tensor-core throughput) - squared norms and the
    post-matmul add/clamp/sqrt arithmetic are always done in fp32 to avoid
    compounding precision loss in the cheap elementwise ops. Pass dtype=torch.float32
    to disable the fp16 path entirely.
    """
    if train_chunk_size is None:
        train_chunk_size = batch_size

    ref_t = torch.from_numpy(ref_np).to(DEVICE).to(dtype)
    ref_sq = (ref_t.float() ** 2).sum(dim=1)
    n_ref = ref_t.shape[0]

    n = len(candidates_np)
    min_dist = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = (
                torch.from_numpy(candidates_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .to(dtype)
            )
            b_sq = (b.float() ** 2).sum(dim=1, keepdim=True)

            running_min = torch.full(
                (b.shape[0],), float("inf"), device=DEVICE, dtype=torch.float32
            )

            for j in range(0, n_ref, train_chunk_size):
                ref_chunk = ref_t[j : j + train_chunk_size]
                ref_sq_chunk = ref_sq[j : j + train_chunk_size]

                cross = (b @ ref_chunk.T).float()
                dist_sq = b_sq + ref_sq_chunk.unsqueeze(0) - 2.0 * cross
                dist_sq.clamp_(min=0)
                dist = dist_sq.sqrt_()

                running_min = torch.minimum(running_min, dist.min(dim=1).values)

                del dist_sq, dist

            min_dist[i : i + batch_size] = running_min.float().cpu().numpy()

            del b, b_sq, running_min

    return min_dist


def pairwise_dist_gpu(a_np, b_np, batch_size, ref_chunk_size=None, dtype=torch.float16):
    """
    Full (len(a_np), len(b_np)) euclidean distance matrix on GPU, chunked over `a_np`
    to bound peak memory. Unlike min_dist_per_row_gpu/min_igs_per_row_gpu this does NOT
    reduce over the reference axis - callers that need the per-pair values (e.g. to
    cache X-distances across AL iterations and combine them with a freshly computed
    Y-distance matrix each iteration) need the full matrix.

    dtype controls the matmul precision only, same fp16-in/fp32-out rationale
    as min_dist_per_row_gpu; the returned matrix is always float32.
    """
    if ref_chunk_size is None:
        ref_chunk_size = batch_size

    b_t = torch.from_numpy(b_np).to(DEVICE).to(dtype)
    b_sq = (b_t.float() ** 2).sum(dim=1)
    n_b = b_t.shape[0]
    n_a = len(a_np)

    out = torch.empty((n_a, n_b), device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, n_a, batch_size):
            a_chunk = (
                torch.from_numpy(a_np[i : i + batch_size]).to(DEVICE, non_blocking=True).to(dtype)
            )
            a_sq = (a_chunk.float() ** 2).sum(dim=1, keepdim=True)

            for j in range(0, n_b, ref_chunk_size):
                b_chunk = b_t[j : j + ref_chunk_size]
                b_sq_chunk = b_sq[j : j + ref_chunk_size]
                cross = (a_chunk @ b_chunk.T).float()
                dist_sq = a_sq + b_sq_chunk.unsqueeze(0) - 2.0 * cross
                dist_sq.clamp_(min=0)
                out[i : i + a_chunk.shape[0], j : j + b_chunk.shape[0]] = dist_sq.sqrt_()

    return out


def min_igs_per_row_gpu(
    X_candidate_np,
    X_train_np,
    Y_candidate_np,
    Y_train_np,
    batch_size,
    train_chunk_size=None,
    dtype=torch.float16,
    profile=False,
):
    """
    Pour chaque ligne candidate, calcule min_m( d_X(n,m) * d_Y(n,m) ) sur les points
    d'entraînement m, en GPU et par batchs. Utilisé pour la stratégie iGS.

    dtype controls the matmul precision only (see min_dist_per_row_gpu's
    docstring for the fp16-in/fp32-out rationale); pass torch.float32 to
    disable it.

    If profile=True, also returns a dict {"t_x": seconds, "t_y": seconds}
    breaking down time spent on the X-distance vs Y-distance chunk
    computations specifically (torch.cuda.synchronize()-guarded for
    accuracy). This adds sync points that prevent CUDA from overlapping
    work across iterations, so only enable it for benchmarking/diagnostics,
    not production runs - it will make the call slower than an unprofiled
    one, not just report on it.
    """
    if train_chunk_size is None:
        train_chunk_size = batch_size

    X_train_t = torch.from_numpy(X_train_np).to(DEVICE).to(dtype)
    X_train_sq = (X_train_t.float() ** 2).sum(dim=1)
    n_train_x = X_train_t.shape[0]

    Y_train_t = torch.from_numpy(Y_train_np).to(DEVICE).to(dtype)
    Y_train_sq = (Y_train_t.float() ** 2).sum(dim=1)
    n_train_y = Y_train_t.shape[0]

    assert (
        n_train_x == n_train_y
    ), "X_train et Y_train doivent avoir le même nombre de lignes (même index)"

    n = len(X_candidate_np)
    min_dxy = np.empty(n, dtype=np.float32)

    t_x_total = 0.0
    t_y_total = 0.0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            bX = (
                torch.from_numpy(X_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .to(dtype)
            )
            bX_sq = (bX.float() ** 2).sum(dim=1, keepdim=True)

            bY = (
                torch.from_numpy(Y_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .to(dtype)
            )
            bY_sq = (bY.float() ** 2).sum(dim=1, keepdim=True)

            running_min = torch.full(
                (bX.shape[0],), float("inf"), device=DEVICE, dtype=torch.float32
            )

            for j in range(0, n_train_x, train_chunk_size):
                Xt_chunk = X_train_t[j : j + train_chunk_size]
                Xsq_chunk = X_train_sq[j : j + train_chunk_size]
                Yt_chunk = Y_train_t[j : j + train_chunk_size]
                Ysq_chunk = Y_train_sq[j : j + train_chunk_size]

                if profile:
                    torch.cuda.synchronize()
                    _t0 = time.time()

                cross_x = (bX @ Xt_chunk.T).float()
                dX_chunk = (bX_sq + Xsq_chunk.unsqueeze(0) - 2.0 * cross_x).clamp_(min=0).sqrt_()

                if profile:
                    torch.cuda.synchronize()
                    t_x_total += time.time() - _t0
                    _t0 = time.time()

                cross_y = (bY @ Yt_chunk.T).float()
                dY_chunk = (bY_sq + Ysq_chunk.unsqueeze(0) - 2.0 * cross_y).clamp_(min=0).sqrt_()

                if profile:
                    torch.cuda.synchronize()
                    t_y_total += time.time() - _t0

                dxy_chunk = dX_chunk * dY_chunk

                running_min = torch.minimum(running_min, dxy_chunk.min(dim=1).values)

                del dX_chunk, dY_chunk, dxy_chunk

            min_dxy[i : i + batch_size] = running_min.float().cpu().numpy()

            del bX, bY, bX_sq, bY_sq, running_min

    if profile:
        return min_dxy, {"t_x": t_x_total, "t_y": t_y_total}
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
        batch_size: int = 8192,
        train_chunk_size: int = None,
        dtype: torch.dtype = torch.float16,
        profile_xy: bool = False,
        **kwargs,
    ):
        """
        Initializes the GreedySamplingSelector.

        Args:
            strategy (str): The greedy sampling strategy. Must be one of 'GSx', 'GSy', or 'iGS'.
            distance (str, optional): The distance metric to use, compatible with `scipy.spatial.distance.cdist`.
            Seed (int, optional): A random seed for reproducibility.
            batch_size (int, optional): Candidate-axis chunk size for the GPU
                distance computation. Was hardcoded to 512, which at
                production-scale candidate pools (millions of rows) means
                hundreds of thousands of tiny kernel launches per select()
                call - launch overhead, not FLOPs, ends up dominating wall
                time. Raise this (memory-permitting) to cut that overhead;
                lower it back down if a chunk stops fitting in GPU memory.
            train_chunk_size (int, optional): Train-axis chunk size. Defaults
                to batch_size if not given.
            dtype (torch.dtype, optional): Precision for the bulk distance
                matmuls (fp16 by default - much higher tensor-core throughput;
                the post-matmul add/clamp/sqrt arithmetic always happens in
                fp32 regardless, so this only affects the expensive part).
                Pass torch.float32 to disable.
            profile_xy (bool, optional): For iGS, report wall-clock time spent
                on the X-distance vs Y-distance computation separately
                (printed by select()). Adds torch.cuda.synchronize() calls
                that block CUDA's ability to overlap work, so this makes
                select() itself slower - only turn it on for benchmarking,
                never for production runs.
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
        self.batch_size = batch_size
        self.train_chunk_size = train_chunk_size if train_chunk_size is not None else batch_size
        self.dtype = dtype
        self.profile_xy = profile_xy

        # Persists across select() calls within one AL run (a fresh selector is
        # created per replication/strategy in LearningProcedure, so this never
        # leaks state across runs). Caches the X-distance matrix (candidates x
        # train) since X is iteration-invariant - only the newly added train
        # points need a distance computed each call, existing columns are
        # reused as-is. Y is NOT cached: it depends on candidate predictions
        # from the freshly retrained model each iteration, so it's recomputed
        # in full every call.
        self._dx_cache = None
        # Full (n_candidate, n_train) fp32 matrix caching is only safe up to a
        # bounded size; beyond this it falls back to the uncached, chunked
        # min-only computation to avoid GPU OOM.
        self._dx_cache_max_bytes = 2 * 1024**3

        # GSx has no evolving second term (no model-dependent Y), so unlike
        # iGS/GSy its score is a per-candidate running MINIMUM that can be
        # cached exactly with O(1) memory per candidate - no dense matrix,
        # no memory cap, no scale limit. See _get_gsx_min_dist.
        self._gsx_cache = None

    def _get_gsx_min_dist(self, candidate_index, train_index, X_candidate_np, X_train_np, batch_size):
        """Incremental running-min X-distance for GSx. Unlike iGS/GSy, GSx's
        score (min distance to the train set) has no second term that changes
        across iterations, so a per-candidate running minimum can be cached
        exactly: each call only needs distances to the newly-added train
        points, merged into the previous minimum via elementwise min - the
        classic incremental-cluster-growth trick. This is a true
        O(n_candidate x k_top) per-call cost with O(n_candidate) memory,
        unlike the dense dX matrix used for iGS which is memory-bounded and
        effectively inert once candidate/train pools get large (see
        _get_dx_matrix)."""
        candidate_labels = candidate_index.tolist()
        train_labels = train_index.tolist()

        if self._gsx_cache is None:
            min_dist = min_dist_per_row_gpu(
                X_candidate_np, X_train_np, batch_size, self.train_chunk_size, self.dtype
            )
            self._gsx_cache = {
                "candidate_labels": candidate_labels,
                "train_labels": list(train_labels),
                "min_dist": min_dist,
            }
            return min_dist

        cache = self._gsx_cache

        # Candidates only ever shrink, train only ever grows (same invariants
        # as _get_dx_matrix - see that docstring).
        old_pos = {lbl: i for i, lbl in enumerate(cache["candidate_labels"])}
        keep_idx = [old_pos[lbl] for lbl in candidate_labels]
        min_dist = cache["min_dist"][keep_idx]

        old_train_set = set(cache["train_labels"])
        new_positions = [i for i, lbl in enumerate(train_labels) if lbl not in old_train_set]

        if new_positions:
            X_train_new_np = X_train_np[new_positions]
            delta = min_dist_per_row_gpu(
                X_candidate_np, X_train_new_np, batch_size, self.train_chunk_size, self.dtype
            )
            min_dist = np.minimum(min_dist, delta)

        self._gsx_cache = {
            "candidate_labels": candidate_labels,
            "train_labels": train_labels,
            "min_dist": min_dist,
        }
        return min_dist

    def _get_dx_matrix(self, candidate_index, train_index, X_candidate_np, X_train_np, batch_size):
        """Returns the full (n_candidate, n_train) X-distance matrix, reusing cached
        columns for train points already seen in a previous call and computing new
        columns only for train points added since then. Returns None if that dense
        matrix would exceed the memory budget - callers must fall back to the
        chunked, non-materializing min_dist_per_row_gpu/min_igs_per_row_gpu instead,
        since (unlike this cache) those never hold more than one chunk at a time."""
        candidate_labels = candidate_index.tolist()
        train_labels = train_index.tolist()

        n_elements = len(candidate_labels) * len(train_labels)
        if n_elements * 4 > self._dx_cache_max_bytes:
            # Too big to hold as a dense cache (and pairwise_dist_gpu would hit the
            # same ceiling, since it materializes the full matrix too) - drop any
            # existing cache so a later, smaller call can re-bootstrap it, and let
            # the caller fall back to a memory-safe path.
            self._dx_cache = None
            return None

        if self._dx_cache is None:
            dX = pairwise_dist_gpu(
                X_candidate_np, X_train_np, batch_size, self.train_chunk_size, self.dtype
            )
            self._dx_cache = {
                "candidate_labels": candidate_labels,
                "train_labels": list(train_labels),
                "dX": dX,
            }
            return dX

        cache = self._dx_cache

        # Candidates only ever shrink (LearningProcedure drops selected rows,
        # never adds any back), so candidate_labels is a subset of the cached
        # ones - prune the cached rows down to the ones still present.
        old_pos = {lbl: i for i, lbl in enumerate(cache["candidate_labels"])}
        keep_idx = [old_pos[lbl] for lbl in candidate_labels]
        dX = cache["dX"][keep_idx, :]

        # Train only ever grows (rows appended) - compute distances for the
        # newly added train points only.
        old_train_set = set(cache["train_labels"])
        new_positions = [i for i, lbl in enumerate(train_labels) if lbl not in old_train_set]

        if new_positions:
            X_train_new_np = X_train_np[new_positions]
            new_cols = pairwise_dist_gpu(
                X_candidate_np, X_train_new_np, batch_size, self.train_chunk_size, self.dtype
            )
            dX = torch.cat([dX, new_cols], dim=1)

        self._dx_cache = {
            "candidate_labels": candidate_labels,
            "train_labels": train_labels,
            "dX": dX,
        }
        return dX

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

        if "subset_rand_candidat" in SimulationConfigInputUpdated["add_useful_params"]:

            n_candidat = SimulationConfigInputUpdated["add_useful_params"]["subset_rand_candidat"]
            if n_candidat is not None:
                # N random seletion
                if len(X_Candidate) > n_candidat:
                    indices_random_iloc = np.random.choice(
                        len(X_Candidate), size=n_candidat, replace=False
                    )
                    X_Candidate = X_Candidate.iloc[indices_random_iloc]

        X_Candidate_f32 = X_Candidate.values.astype(np.float32)
        X_Train_f32 = X_Train.values.astype(np.float32)

        select_ytrain_cols = None

        batch_size = self.batch_size

        # print(f"\t+++ GreedySampling #1 : {time.time() - StartTime} +++")

        ## 2. Prediction on candidates (only needed for GSy / iGS) ##
        StartTime = time.time()

        Predictions = None
        if self.strategy in ["GSy", "iGS"]:
            if (
                SimulationConfigInputUpdated is not None
                and SimulationConfigInputUpdated.get("hl_trainer") is not None
            ):
                cached = SimulationConfigInputUpdated.get("_cached_candidate_predictions")
                cached_cols = SimulationConfigInputUpdated.get("all_reg_cols")

                if (
                    cached is not None
                    and cached_cols is not None
                    and cached["labels"].equals(X_Candidate.index)
                ):
                    # FullPoolErrorFunction already ran this exact predict() call
                    # earlier in this same iteration (same model, same candidate
                    # set) - reuse it instead of paying for it twice.
                    select_ytrain_cols = cached_cols
                    Predictions = cached["y_pred_pd"][select_ytrain_cols].values
                else:
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
            final_scores = self._get_gsx_min_dist(
                df_Candidate.index, df_Train.index, X_Candidate_f32, X_Train_f32, batch_size
            )

        elif self.strategy == "GSy":
            final_scores = min_dist_per_row_gpu(
                pred_vals, y_train_values, batch_size, self.train_chunk_size, self.dtype
            )

        elif self.strategy == "iGS":
            if self.profile_xy:
                torch.cuda.synchronize()
                _t0 = time.time()
            dX_matrix = self._get_dx_matrix(
                df_Candidate.index, df_Train.index, X_Candidate_f32, X_Train_f32, batch_size
            )
            if self.profile_xy:
                torch.cuda.synchronize()
                t_x = time.time() - _t0  # cache lookup/prune + any new columns, or a quick None

            if dX_matrix is None:
                # No cache benefit at this size - X and Y are computed together,
                # chunk by chunk, inside this one call (see min_igs_per_row_gpu).
                if self.profile_xy:
                    final_scores, xy_timing = min_igs_per_row_gpu(
                        X_Candidate_f32, X_Train_f32, pred_vals, y_train_values,
                        batch_size, self.train_chunk_size, self.dtype, profile=True,
                    )
                    print(
                        f"\t+++ GreedySampling X/Y split (uncached, dX_matrix lookup {t_x:.3f}s): "
                        f"X={xy_timing['t_x']:.3f}s  Y={xy_timing['t_y']:.3f}s +++"
                    )
                else:
                    final_scores = min_igs_per_row_gpu(
                        X_Candidate_f32, X_Train_f32, pred_vals, y_train_values,
                        batch_size, self.train_chunk_size, self.dtype,
                    )
            else:
                # Y depends on this iteration's freshly retrained model (candidate
                # predictions change every call), so it's never cached - recomputed
                # in full and combined with the cached/incremental X matrix above.
                if self.profile_xy:
                    torch.cuda.synchronize()
                    _t0 = time.time()
                dY_matrix = pairwise_dist_gpu(
                    pred_vals, y_train_values, batch_size, self.train_chunk_size, self.dtype
                )
                if self.profile_xy:
                    torch.cuda.synchronize()
                    t_y = time.time() - _t0
                    print(
                        f"\t+++ GreedySampling X/Y split (cached): "
                        f"X={t_x:.3f}s (cache hit)  Y={t_y:.3f}s (always recomputed) +++"
                    )
                final_scores = (dX_matrix * dY_matrix).min(dim=1).values.float().cpu().numpy()

        # print(f"\t+++ GreedySampling final_scores : {time.time() - StartTime} +++")

        ## 4. Sequential (greedy) top-k selection with incremental score updates ##
        # True greedy sampling (Wu, Lin & Huang 2018) picks one candidate at a time and
        # updates the reference set before scoring the next pick, using the picked
        # candidate's own prediction as a proxy label since it isn't labeled yet.
        # Picking a whole batch off a single distance snapshot (the previous behavior
        # here) can select several mutually-close "far" points instead of a diverse
        # batch. Recomputing full distances against the whole reference set for every
        # pick would cost O(k * N * M); instead each remaining candidate only needs its
        # distance to the ONE just-picked point, merged into a running min, since
        # min(d_to_old_ref, d_to_new_point) == min(d_to_old_ref U {new_point}).
        #
        # This delta step deliberately does NOT reuse min_dist_per_row_gpu /
        # min_igs_per_row_gpu: those are built for a (possibly large) M-row
        # reference set, so every call round-trips the full remaining candidate
        # array through host memory and recomputes its squared norms from
        # scratch. Called once per round that's a real, avoidable cost - paid
        # twice per round for iGS, since it repeats that host-transfer/norm tax
        # for both the X and Y branches. Here the reference is always exactly
        # one row, so candidates are kept resident on the GPU across the whole
        # round loop and each delta is a plain (N, D) . (D,) distance-to-a-point,
        # no chunking or host round-trip required.
        StartTime = time.time()

        top_k_number = self.k_top_candidate
        if len(final_scores) < self.k_top_candidate:
            top_k_number = len(final_scores)

        if top_k_number <= 1:
            # single pick - no reference-set update needed, skip GPU residency setup
            best_candidate_iloc = (
                np.array([int(np.argmax(final_scores))])
                if top_k_number == 1
                else np.array([], dtype=int)
            )
        else:
            remaining_iloc = torch.arange(len(final_scores), device=DEVICE)
            running_scores = torch.from_numpy(final_scores).to(DEVICE)

            X_remaining = torch.from_numpy(X_Candidate_f32).to(DEVICE)
            X_sq = (X_remaining**2).sum(dim=1)

            if pred_vals is not None:
                y_pred_remaining = torch.from_numpy(pred_vals).to(DEVICE)
                y_sq = (y_pred_remaining**2).sum(dim=1)

            selected_iloc = []
            for round_idx in range(top_k_number):
                pick_pos = int(torch.argmax(running_scores))
                selected_iloc.append(int(remaining_iloc[pick_pos]))

                if round_idx == top_k_number - 1:
                    break  # last pick - no further round needs an updated reference set

                if self.strategy in ("GSx", "iGS"):
                    new_ref_X = X_remaining[pick_pos]
                    new_ref_X_sq = X_sq[pick_pos]
                if self.strategy in ("GSy", "iGS"):
                    new_ref_Y = y_pred_remaining[pick_pos]
                    new_ref_Y_sq = y_sq[pick_pos]

                keep = torch.ones(remaining_iloc.shape[0], dtype=torch.bool, device=DEVICE)
                keep[pick_pos] = False
                remaining_iloc = remaining_iloc[keep]
                running_scores = running_scores[keep]
                X_remaining = X_remaining[keep]
                X_sq = X_sq[keep]
                if pred_vals is not None:
                    y_pred_remaining = y_pred_remaining[keep]
                    y_sq = y_sq[keep]

                if self.strategy == "GSx":
                    delta = (
                        (X_sq + new_ref_X_sq - 2.0 * (X_remaining @ new_ref_X))
                        .clamp_(min=0)
                        .sqrt_()
                    )
                elif self.strategy == "GSy":
                    delta = (
                        (y_sq + new_ref_Y_sq - 2.0 * (y_pred_remaining @ new_ref_Y))
                        .clamp_(min=0)
                        .sqrt_()
                    )
                else:  # iGS
                    dX = (
                        (X_sq + new_ref_X_sq - 2.0 * (X_remaining @ new_ref_X))
                        .clamp_(min=0)
                        .sqrt_()
                    )
                    dY = (
                        (y_sq + new_ref_Y_sq - 2.0 * (y_pred_remaining @ new_ref_Y))
                        .clamp_(min=0)
                        .sqrt_()
                    )
                    delta = dX * dY

                running_scores = torch.minimum(running_scores, delta)

            best_candidate_iloc = np.array(selected_iloc)

        # print(f"\t+++ GreedySampling sequential selection : {time.time() - StartTime} +++")

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
