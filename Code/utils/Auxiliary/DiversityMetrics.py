# Import necessary library
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

def DiversityMetricsFunction(df_Candidate, df_Train, k=10):

    ### Set Up ###
    X_Candidate = df_Candidate.loc[:, df_Candidate.columns!= "Y"]
    X_Train = df_Train.loc[:,df_Train.columns!= "Y"]

    ### Diversity Metric ###
    d_nmX = distance.cdist(X_Candidate, X_Train, metric = "euclidean")
    d_nX = d_nmX.min(axis=1)

    ### Density Metrics ###
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is counted
    knn.fit(X_Candidate)
    distances, indices = knn.kneighbors(X_Candidate)
    similarities = np.exp(-distances[:, 1:])  # Exclude the first one (self)
    DensityScores = np.mean(similarities, axis=1)

    # ### Clustering ###
    # cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
    # ClusterLabels = cluster.fit_predict(X_Candidate)

    ### Store ###
    df_Candidate["DiversityScores"] = d_nX
    # df_Candidate["ClusterLabels"] = ClusterLabels
    df_Candidate["DensityScores"] = DensityScores

    ### Return ###
    return df_Candidate