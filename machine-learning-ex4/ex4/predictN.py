from sigmoid import *


def predict(theta1, theta2, X):
    # Useful values
    m = X.shape[0]
    a1 = np.c_[np.ones(m), X]  # 输入层

    z2 = a1.dot(theta1.T)
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)  # a2 = sigmoid(z2)  # 隐藏层
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)  # 输出层

    p = np.argmax(a3, axis=1) + 1

    return p


