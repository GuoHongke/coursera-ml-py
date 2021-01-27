from sigmoid import sigmoid


def sigmoid_gradient(z):
    a = sigmoid(z)
    g = a * (1 - a)
    # ===================== Your Code Here =====================
    # Instructions : Compute the gradient of the sigmoid function evaluated at
    #                each value of z (z can be a matrix, vector or scalar)
    #


    # ===========================================================

    return g


def sigmoid_gradient_a(a):
    # 此处直接传a，介绍预算成本
    g = a * (1 - a)
    return g
