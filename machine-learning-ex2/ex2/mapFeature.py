def map_feature(X):
    degree = 6
    m, n = X.shape
    X.insert(0, 'Ones', 1)
    for i in range(degree + 1):
        for j in range(degree - i + 1):
            if i + j < n:
                continue
            X['Test{}{}'.format(i, j)] = X[['Test1', 'Test2']].apply(lambda x: (x['Test1']**i) * (x['Test2']**j), axis=1)

    return X
