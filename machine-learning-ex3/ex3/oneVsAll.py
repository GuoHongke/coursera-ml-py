import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    initial_theta = np.zeros(n + 1)
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]
    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        # ===================== Your Code Here =====================
        # Instructions : You should complete the following code to train num_labels
        #                logistic regression classifiers with regularization
        #                parameter lambda
        #
        #
        # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
        #       whether the ground truth is true/false for this class
        #
        # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
        #       function. It is okay to use a for-loop (for c in range(num_labels) to
        #       loop over the different classes
        #
        y_i = np.array([1 if _y == (10 if i == 0 else i) else 0 for _y in y[:, 0]])
        res = opt.minimize(fun=lCF.lr_cost_function, x0=initial_theta, args=(X, y_i.reshape((len(y_i), 1)), lmd),
                           method='BFGS', jac=lCF.lr_grad_function)
        all_theta[i:i+1, :] = res.x
        # ============================================================
        print('Done')
    return all_theta
