import numpy as np


def feature_normalize(X, y):
    # You need to set these values correctly
    x_mean = X.mean()
    y_mean = y.mean()
    x_std = X.std()
    y_std = y.std()

    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    mu = np.array([x_mean['Size'], x_mean['Bedrooms'], y_mean['Price']])
    sigma = np.array([x_std['Size'], x_std['Bedrooms'], y_std['Price']])
    # ===================== Your Code Here =====================
    # Instructions : First, for each feature dimension, compute the mean
    #                of the feature and subtract it from the dataset,
    #                storing the mean value in mu. Next, compute the
    #                standard deviation of each feature and divide
    #                each feature by its standard deviation, storing
    #                the standard deviation in sigma
    #
    #                Note that X is a 2D array where each column is a
    #                feature and each row is an example. You need
    #                to perform the normalization separately for
    #                each feature.
    #
    # Hint: You might find the 'np.mean' and 'np.std' functions useful.
    #       To get the same result as Octave 'std', use np.std(X, 0, ddof=1)
    #



    # ===========================================================

    return X_norm, y_norm, mu, sigma
