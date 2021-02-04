import numpy as np


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()

    sim = np.exp(-np.power((x1 - x2), 2).sum() / (sigma ** 2 * 2))

    # ===================== Your Code Here =====================
    # Instructions : Fill in this function to return the similarity between x1
    #                and x2 computed using a Gaussian kernel with bandwith sigma
    #


    # ==========================================================

    return sim

