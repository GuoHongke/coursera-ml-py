import numpy as np
from scipy import linalg


def pca(X):
    # Useful values
    (m, n) = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ===================== Your Code Here =====================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the 'scipy.linalg.svd' function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    # 
    # Hint: Take a look at full_matrices, compute_uv parameters for the svd function
    #
    
    Sigma = X.T @ X / m
    U, S, V = linalg.svd(Sigma)
    # ==========================================================

    return U, S
