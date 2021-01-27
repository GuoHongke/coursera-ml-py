import numpy as np


def rand_initialization(l_in, l_out):
    # You need to return the following variable correctly
    epsilon_init = 0.12
    w = np.random.rand(l_in + 1, l_out) * 2 * epsilon_init - epsilon_init

    # ===================== Your Code Here =====================
    # Instructions : Initialize w randomly so that we break the symmetry while
    #                training the neural network
    #
    # Note : The first column of w corresponds to the parameters for the bias unit
    #


    # ===========================================================

    return w
