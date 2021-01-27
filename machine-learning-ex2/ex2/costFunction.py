import numpy as np
from sigmoid import sigmoid


def cost_function(theta, X, y):
    m = X.shape[0]
    # You need to return the following values correctly
    g = np.array(sigmoid(X.dot(theta)))
    cost = (y - 1) * (np.log(1 - g)) - y * (np.log(g))
    grad = (1 / m) * X.T.dot(g - y)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    # ===========================================================
    return np.mean(cost.values), grad.values.reshape(len(theta),)
