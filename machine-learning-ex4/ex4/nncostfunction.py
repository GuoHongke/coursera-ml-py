import numpy as np
from feed_forward import feed_forward, deserialize_t
from sigmoidgradient import sigmoid_gradient_a, sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):

    # Useful value
    m = y.size

    theta1, theta2 = deserialize_t(nn_params, input_layer_size, hidden_layer_size, num_labels)
    _, _, _, _, h = feed_forward(theta1, theta2, X)

    ys = extend_y(y, num_labels)
    cost = np.sum(-ys * np.log(h) - (1 - ys) * np.log(1 - h))

    return cost / m + nn_cost_function_reg(theta1, theta2, m, lmd)


def nn_cost_function_reg(theta1, theta2, m, lmd):
    all_theta = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))
    return lmd * all_theta / m / 2


def nn_grad_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1, theta2 = deserialize_t(nn_params, input_layer_size, hidden_layer_size, num_labels)
    a1, z2, a2, z3, h = feed_forward(theta1, theta2, X)  # a1(5000, 401) a2(5000, 26), h(5000, 10)
    # Useful value
    m = y.size
    ys = extend_y(y, num_labels)
    d3 = h - ys  # (5000, 10)
    d2 = d3 @ theta2 * sigmoid_gradient_a(a2)  # (5000, 26)

    theta2_grad = d3.T.dot(a2) / m + nn_grad_function_reg(theta2, lmd, m)
    theta1_grad = d2.T.dot(a1)[1:, :] / m + nn_grad_function_reg(theta1, lmd, m)

    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])
    return grad


def nn_grad_function_reg(theta, lmd, m):
    theta[:, 0] = 0
    return lmd * theta / m


def extend_y(y, num_labels):
    m = y.size
    ys = np.zeros((m, num_labels))
    for i, j in enumerate(y):
        ys[i:i + 1, j - 1] = 1
    return ys
