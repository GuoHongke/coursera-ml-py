import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    def lambda_f(x):
        return 1 if x[0] >= 0 else 0
    p = X.dot(theta)
    p[1] = p[[0]].apply(lambda_f, axis=1)
    # p = X.dot(theta).apply(lambda x:)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #


    # ===========================================================

    return p.iloc[:, -1]
