import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        """
        get the means for each class of data
        there are two classes: 0 and 1

        first, we add intercepts to the data

        Sw is the within-class covariance matrix
        for Sw, we can just sum over each term of the outer product of the difference of the individual line of data and the mean of its respective class
        then, using the formula from lecture, W is solved as Sw^-1 x (mean2 - mean1) (for two classes)
        """
        classes = [0, 1]

        # add the bias
        X_rows, X_cols = X.shape
        bias = np.ones([X_rows, 1])
        M = np.concatenate((bias, X), axis = 1)
        M_mean = np.mean(M, axis = 0)

        M_rows, M_cols = M.shape

        Sw = np.zeros([M_cols, M_cols])
        means = []

        for class_val in classes:
            # get the matrix with the specific class value
            Mi = M[y == class_val]
            Mi_rows, Mi_cols = Mi.shape

            # the means for (x1, x2) are located on their respective columns
            Mi_mean = np.mean(Mi, axis = 0)
            means.append(Mi_mean)

            for data in Mi:
                # inner summation portion of the formula; so if we add everything, this should be the double summation
                Sw += np.outer(data - Mi_mean, data - Mi_mean)

        m1 = means[0]
        m2 = means[1]

        # definition as stated in lecture
        self.W = np.linalg.pinv(Sw).dot(m2 - m1)

    def predict(self, X):
        """
        since we want to fit y into two classes, we need to classify each y value in y_pred as y_class
        classification is done by thresholds described in lecture: f(x) >= lambda (y >= lambda_val in our case)
        these are then compared to the true values to get the error rate
        """
        # adding bias first
        X_rows, X_cols = X.shape
        bias = np.ones([X_rows, 1])
        M = np.concatenate((bias, X), axis = 1)

        # projecting data on W to get the predicted y
        y_pred = M.dot(self.W)
        y_class = y_pred >= self.lambda_val

        return y_class
