import numpy as np


def linear_reg_cost_function(theta, X, y, lmd):
    # Initialize some useful values
    m = X.shape[0]

    # You need to return the following variables correctly
    theta = theta.reshape((len(theta), 1))
    err = X @ theta - y
    # cost = ( + lmd * np.sum(np.power(theta, 2))) / (m * 2)
    cost = (np.sum(np.power(err, 2)) + lmd * np.sum(np.power(theta[1:], 2))) / (2 * m)

    return cost


def linear_reg_grad_function(theta, X, y, lmd):
    m = X.shape[0]
    theta = theta.reshape((len(theta), 1))
    err = X @ theta - y
    theta_tmp = theta.copy()
    theta_tmp[0] = 0
    grad = (X.T @ err + lmd * theta) / m
    return grad.reshape((len(grad)))

