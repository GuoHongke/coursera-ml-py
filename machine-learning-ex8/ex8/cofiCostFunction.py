import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    cost = (np.power((X @ theta.T)[np.nonzero(R)] - Y[np.nonzero(R)], 2).sum() +
            lmd * np.power(X, 2).sum() + lmd * (np.power(theta, 2).sum())) / 2

    return cost


def cofi_gred_function(params, Y, R, num_users, num_movies, num_features, lmd):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    # You need to set the following values correctly.
    X_1 = X @ theta.T
    X_1[R == 0] = 0
    X_grad = (X_1 - Y) @ theta + lmd * X

    theta_grad = (X_1 - Y).T @ X + lmd * theta

    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten()))

    return grad