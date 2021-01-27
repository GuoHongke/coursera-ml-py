import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    # =========================================================
    inner = np.power(((np.array(X).dot(theta)) - y), 2)
    return np.sum(inner) / (2 * len(X))
