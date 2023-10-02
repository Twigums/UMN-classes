import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        """
        solution to the ridge regression is W = (X^T x X + lambda x I) ^ -1 x X^T x y
        """
        # add bias first
        X_rows, X_cols = X.shape
        bias = np.ones([X_rows, 1])
        M = np.concatenate((bias, X), axis = 1)
        M_rows, M_cols = M.shape

        I = np.identity(M_cols)

        # the explicit solution as described above
        self.W = np.linalg.inv(M.T.dot(M) + self.lambda_val * I).dot(M.T).dot(y)

    def predict(self, X):
        # add bias like above
        X_rows, X_cols = X.shape
        bias = np.ones([X_rows, 1])
        M = np.concatenate((bias, X), axis = 1)
        M_rows, M_cols = M.shape

        # prediction is just the matrix M x W
        y_pred = M.dot(self.W)

        return y_pred
