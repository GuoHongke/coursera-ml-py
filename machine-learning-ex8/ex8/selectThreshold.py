import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def select_threshold(X, Xval, yval):
    f1 = 0

    # You have to return these values correctly
    best_eps = 0
    best_f1 = 0
    mu = np.mean(X, axis=0)
    sigma2 = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, sigma2)

    pval = multi_normal.pdf(Xval)

    for epsilon in np.linspace(np.min(pval), np.max(pval), num=10000):
        # ===================== Your Code Here =====================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note : You can use predictions = pval < epsilon to get a binary vector
        #        of False(0)'s and True(1)'s of the outlier predictions
        #
        y_pred = (pval < epsilon).astype(int)
        f1 = f1_score(yval, y_pred)

        # ==========================================================

        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1, multi_normal
