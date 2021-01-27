import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotData import *
from mapFeature import *


def plot_decision_boundary(theta, X):
    if X.shape[1] <= 3:
        # Only need two points to define a line, so choose two endpoints
        plot_x = np.linspace(0, 100, 100)  # 等差数列
        plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y)
        plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc=1)
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        x = np.linspace(-1, 1.5, 50)
        y = np.linspace(-1, 1.5, 50)
        xs, ys = np.meshgrid(x, y)
        z = np.zeros((x.size, y.size))

        # Evaluate z = theta*x over the grid
        test1 = []
        test2 = []
        for i in range(x.size):
            test1 += [x[i] for _ in range(y.size)]
            test2 += list(y)
        test_X = pd.DataFrame({'Test1': test1, 'Test2': test2})
        res = map_feature(test_X).dot(theta)
        for i in range(x.size):
            for j in range(y.size):
                z[i, j] = res[i * x.size + j]

        # Evaluate z = theta*x over the grid
        # for i in range(0, x.size):
        #     for j in range(0, y.size):
        #         z[i, j] = map_feature(pd.DataFrame({'Test1': [x[i]], 'Test2': [y[j]]})).dot(theta)

        z = z.T
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(xs, ys, z, levels=[0], colors='r')
