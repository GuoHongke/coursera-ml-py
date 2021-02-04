import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    plt.figure()
    pos_X = X.take([i for i in range(len(y)) if y[i] == 1], axis=0)
    neg_X = X[np.ix_([i for i in range(len(y)) if y[i] == 0], [0, 1])]
    fig, ax = plt.subplots()
    ax.scatter(pos_X[:, 0], pos_X[:, 1], c='b', marker='+')
    ax.scatter(neg_X[:, 0], neg_X[:, 1], c='y', marker='o')
