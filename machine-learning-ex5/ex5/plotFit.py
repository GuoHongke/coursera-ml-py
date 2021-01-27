import matplotlib.pyplot as plt
import numpy as np
import polyFeatures as pf


def plot_fit(min_x, max_x, mu, sigma, theta, p, lmd):
    x = np.arange(min_x - 15, max_x + 15, 0.1)
    x = x.reshape((len(x), 1))
    X_poly = pf.poly_features(x, p)
    X_poly -= mu
    X_poly /= sigma

    X_poly = np.c_[np.ones(x.size), X_poly]

    plt.plot(x, np.dot(X_poly, theta))
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water folowing out of the dam (y)')
    plt.ylim([-20, 60])
    plt.title('Polynomial Regression Fit (lambda = {})'.format(lmd))
    plt.show()

