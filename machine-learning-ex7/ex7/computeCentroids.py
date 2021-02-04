import numpy as np


def compute_centroids(X, idx, K):
    # Useful values
    (m, n) = X.shape

    # You need to return the following variable correctly.
    centroids = np.zeros((K, n))

    # ===================== Your Code Here =====================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids[i]
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    for i in range(K):
        idx_i = [index for index in range(m) if idx[index] == i]
        X_i = X[idx_i, :]
        centroids[i] = np.sum(X_i, axis=0) / float(len(idx_i))
    # ==========================================================

    return centroids
