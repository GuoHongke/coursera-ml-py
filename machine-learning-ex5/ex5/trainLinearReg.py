import numpy as np
import linearRegCostFunction as lrcf
import scipy.optimize as opt


def train_linear_reg(X, y, lmd):
    initial_theta = np.ones(X.shape[1])

    res = opt.minimize(lrcf.linear_reg_cost_function, initial_theta, args=(X, y, lmd), method='TNC',
                       jac=lrcf.linear_reg_grad_function)

    return res.x
