import numpy as np


def normal_eqn(X, y):
    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return theta
