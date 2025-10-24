# Summary: Implements iRDM (Liu et al., 2021).
#          Unsupervised, batch-mode ALR that balances Representativeness (R) and Diversity (D)
#          via RD initialization and iterative per-cluster maximization of D - R.

### Libraries ###
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from utils.Auxiliary.DataFrameUtils import get_features_and_target 

class IRDMSelector:

    ### Initialize ###
    def __init__(self, 
                 BatchSize: int = 10, 
                 Seed: int = None, 
                 cmax: int = 5,
                 **kwargs):
        self.BatchSize = int(BatchSize)
        self.Seed = Seed
        self.cmax = int(cmax)

    def _zscore(self, X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (X - mu) / sd

    ### Select Observation ###
    def select(self, 
               df_Candidate: pd.DataFrame, 
               Model=None, 
               df_Train: pd.DataFrame = None,
               auxiliary_columns: list = None,
               **kwargs) -> dict:
        if df_Candidate is None or df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Set up candidate features ##
        X_Cand, _ = get_features_and_target(df=df_Candidate, target_column_name="Y")
        X = X_Cand.values
        N = X.shape[0]
        M = min(max(1, self.BatchSize), N)

        ## Standardize features (improves distance quality) ##
        Xs = self._zscore(X)

        ## RD Initialization via k-means (k = M), pick nearest to each centroid ##
        kmeans = KMeans(n_clusters=M, n_init=10, random_state=self.Seed)
        labels = kmeans.fit_predict(Xs)
        centroids = kmeans.cluster_centers_

        selected_idx = []
        for m in range(M):
            members = np.where(labels == m)[0]
            if members.size == 0:
                # Fallback: pick global nearest to this centroid
                d2 = cdist(centroids[m][None, :], Xs, metric="sqeuclidean").reshape(-1)
                chosen = int(np.argmin(d2))
            else:
                d2 = cdist(centroids[m][None, :], Xs[members], metric="sqeuclidean").reshape(-1)
                chosen = int(members[np.argmin(d2)])
            selected_idx.append(chosen)
        selected_idx = np.array(selected_idx, dtype=int)

        ## Precompute pairwise Euclidean distances among all candidates ##
        PD = cdist(Xs, Xs, metric="euclidean")

        def optimize_once(sel_idx: np.ndarray) -> tuple[np.ndarray, bool]:
            changed = False
            for m in range(M):
                cluster_members = np.where(labels == m)[0]
                if cluster_members.size == 0:
                    continue

                # R(x): average distance to other points in the same cluster
                if cluster_members.size > 1:
                    subD = PD[np.ix_(cluster_members, cluster_members)]
                    R = subD.sum(axis=1) / (cluster_members.size - 1)
                else:
                    R = np.zeros(1, dtype=float)

                # D(x): min distance to the other M-1 selected points (excluding this slot)
                others = np.delete(sel_idx, m)
                if others.size > 0:
                    D = PD[cluster_members][:, others].min(axis=1)
                else:
                    D = np.full(cluster_members.size, np.inf)

                score = D - R
                best_local = cluster_members[int(np.argmax(score))]
                if best_local != sel_idx[m]:
                    sel_idx[m] = best_local
                    changed = True
            return sel_idx, changed

        ## Iterative R–D maximization with cycle detection (paper-style) ##
        seen = set()
        iters = 0
        while iters < self.cmax:
            # break if we've seen this set before (sorted for order-invariance)
            key = tuple(sorted(int(i) for i in selected_idx))
            if key in seen:
                break
            seen.add(key)

            selected_idx, changed = optimize_once(selected_idx)
            if not changed:
                break
            iters += 1

        ## Output ##
        rec_indices = df_Candidate.iloc[selected_idx].index.astype(float).tolist()
        return {"IndexRecommendation": rec_indices}