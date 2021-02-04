import numpy as np


def multivariate_gaussian(X, mu, sigma2):
    k = mu.size
    x = X - mu
    exp = - (1 / 2) * (x @ np.linalg.pinv(sigma2) @ x.T)
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(sigma2) ** (-0.5) * exp


    return p
