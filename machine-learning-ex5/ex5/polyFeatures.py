import numpy as np


def poly_features(X, p):
    # You need to return the following variable correctly.
    # ===================== Your Code Here =====================
    # Instructions : Given a vector X, return a matrix X_poly where the p-th
    #                column of X contains the values of X to the p-th power.
    #
    if p == 1:
        return X

    X_poly = np.zeros((X.shape[0], p))
    X_poly[:, 0] = X[:, 0]
    for i in range(1, p):
        X_poly[:, i] = X_poly[:, i - 1] * X[:, 0]

    # ==========================================================

    return X_poly