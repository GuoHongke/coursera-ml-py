#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def feed_forward(theta1, theta2, X):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    # 前向传播算法

    m = X.shape[0]
    a1 = np.c_[np.ones(m), X]
    z2 = a1.dot(theta1.T)
    a2 = np.c_[np.ones(m), sigmoid(z2)]
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def deserialize_t(nn_params, input_layer_size, hidden_layer_size, num_labels):
    # 把展开后的theta重新加载成原来的格式
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    return theta1, theta2
