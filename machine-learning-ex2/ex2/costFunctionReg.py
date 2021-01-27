import numpy as np
from sigmoid import sigmoid


def cost_function_reg(theta, X, y, lmd):
    m = len(y)
    theta_len = len(theta)
    theta = theta.reshape((theta_len, 1))
    # You need to return the following values correctly
    g = np.array(sigmoid(X.dot(theta)))
    cost = (np.sum(-y * np.log(g) - (1 - y) * np.log(1 - g)) + lmd / 2 * np.sum(np.power(theta[1:], 2))) / m
    grad = (1 / m) * X.T.dot(g - y) + lmd / m * theta[1:]

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    return cost, grad.values.reshape(theta_len,)
