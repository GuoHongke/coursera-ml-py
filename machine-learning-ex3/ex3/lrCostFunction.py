import numpy as np
from sigmoid import sigmoid


def lr_cost_function(theta, X, y, lmd):
    m = len(y)
    # You need to return the following values correctly
    theta = theta.reshape((len(theta), 1))
    g = np.array(sigmoid(X.dot(theta)))
    cost = (np.sum(-y * np.log(g) - (1 - y) * np.log(1 - g)) + lmd / 2 * np.sum(np.power(theta[1:], 2))) / m

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    # ===========================================================
    return cost


def lr_grad_function(theta, X, y, lmd):
    m = len(y)
    theta_len = len(theta)
    theta = theta.reshape((theta_len, 1))
    g = np.array(sigmoid(X.dot(theta)))
    grad = (1 / m) * X.T.dot(g - y) + lmd / m * theta[1:]
    return grad.reshape(theta_len,)
