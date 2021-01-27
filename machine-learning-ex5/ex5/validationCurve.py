import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def validation_curve(X, y, Xval, yval):
    # Selected values of lambda (don't change this)
    lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = []
    error_val = []

    # ===================== Your Code Here =====================
    # Instructions : Fill in this function to return training errors in
    #                error_train and the validation errors in error_val. The
    #                vector lambda_vec contains the different lambda parameters
    #                to use for each calculation of the errors, i.e,
    #                error_train[i], and error_val[i] should give
    #                you the errors obtained after training with
    #                lmd = lambda_vec[i]
    #
    for lmd in lambda_vec:
        theta_l = tlr.train_linear_reg(X, y, lmd)
        error_train.append(lrcf.linear_reg_cost_function(theta_l, X, y, lmd))
        error_val.append(lrcf.linear_reg_cost_function(theta_l, Xval, yval, lmd))

    # ==========================================================

    return lambda_vec, error_train, error_val
