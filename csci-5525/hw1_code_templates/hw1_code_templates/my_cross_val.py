import numpy as np

def my_cross_val(model, loss_func, X, y, k=10):
    # will have the final list of losses from each fold
    loss = []

    X_rows, X_cols = X.shape

    # include nonzero intercepts with X
    bias = np.ones([X_rows, 1])
    X = np.concatenate((bias, X), axis = 1)

    fold_size = X_rows // k

    """
    each fold will have a test set the size of fold_size and the training data will exclude the test data
    the last fold will have the remaining set of data if the size of the dataset is not perfectly divisible by k
    """
    for i in range(k):

        # last set should contain the rest of the data
        if i == k - 1:
            start = i * fold_size
            end = X_rows

        else:
            start = i * fold_size
            end = (i + 1) * fold_size

        X_test = X[start: end]
        X_train = np.delete(X, slice(start, end), axis = 0)
        y_test = y[start: end]
        y_train = np.delete(y, slice(start, end))

        # fitting and predicting with the inputted model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # preparations for calculating errors before appending the errors into the loss list
        y_size = len(y_pred)
        error_sum = 0

        # mse error calculations from equation (1)
        if loss_func == "mse":
            for j in range(y_size):
                error_sum += (y_test[j] - y_pred[j]) ** 2

            error = error_sum / y_size

        # error rate calculations from equation (2)
        if loss_func == "err_rate":
            error_vec = y_test != y_pred
            error = sum(error_vec) / y_size

        loss.append(error)

    return loss
