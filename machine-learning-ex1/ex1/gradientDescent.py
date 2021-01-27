import numpy as np
from computeCost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values, can also deal with multi theta(>2)
    m = len(X)
    theta_len = len(theta)
    J_history = np.zeros(num_iters)
    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

        inner = np.array(X).dot(theta) - y
        for j in range(theta_len):
            theta[j, 0] = theta[j, 0] - (alpha / m * (np.sum(inner.multiply(np.array(X.iloc[:, j:j+1])))))['Price']
        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
