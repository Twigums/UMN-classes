################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################

"""
to generalize this and make sure I choose the optimal lambda, I will use variables to check the lowest means
these values will never be -1 when testing since the error >= 0
"""
rr_lowest_mean = -1
lasso_lowest_mean = -1

lambda_rr_opt = 0
lambda_lasso_opt = 0

for lambda_val in lambda_vals:

    # instantiate ridge regression object
    rr_model = MyRidgeRegression(lambda_val)

    # call to your CV function to compute mse for each fold
    rr_mse_vals = my_cross_val(rr_model, "mse", X, y, k = 10)
    rr_mse_mean = np.mean(rr_mse_vals)
    rr_mse_std = np.std(rr_mse_vals)

    # print mse from CV
    print("MSE from Ridge Regression: ", rr_mse_vals)
    print("Average MSE from Ridge Regression: ", rr_mse_mean)
    print("Standard Deviation of MSE from Ridge Regression: ", rr_mse_std, "\n")

    # instantiate lasso object
    lasso_model = Lasso(lambda_val)

    # call to your CV function to compute mse for each fold
    lasso_mse_vals = my_cross_val(lasso_model, "mse", X, y, k = 10)
    lasso_mse_mean = np.mean(lasso_mse_vals)
    lasso_mse_std = np.std(lasso_mse_vals)

    # print mse from CV
    print("MSE from Lasso: ", lasso_mse_vals)
    print("Average MSE from Lasso: ", lasso_mse_mean)
    print("Standard Deviation of MSE from Lasso: ", lasso_mse_std, "\n")

    # chooses the best lambdas based on the respective means
    if rr_lowest_mean == -1:
        rr_lowest_mean = rr_mse_mean
        lambda_rr_opt = lambda_val

    elif rr_lowest_mean > rr_mse_mean:
        rr_lowest_mean = rr_mse_mean
        lambda_rr_opt = lambda_val

    if lasso_lowest_mean == -1:
        lasso_lowest_mean = lasso_mse_mean
        lambda_lasso_opt = lambda_val

    elif lasso_lowest_mean > lasso_mse_mean:
        lasso_lowest_mean = lasso_mse_mean
        lambda_lasso_opt = lambda_val

# instantiate ridge regression and lasso objects for best values of lambda
print("Lowest Ridge Regression MSE mean at lambda_val = ", lambda_rr_opt, " with mean = ", rr_lowest_mean)
print("Lowest Lasso MSE mean at lambda_val = ", lambda_lasso_opt, " with mean = ", lasso_lowest_mean)

rr_model_opt = MyRidgeRegression(lambda_rr_opt)
lasso_model_opt = Lasso(lambda_lasso_opt)

# fit models using all training data
rr_model_opt.fit(X_train, y_train)
lasso_model_opt.fit(X_train, y_train)

# predict on test data
y_pred_rr = rr_model_opt.predict(X_test)
y_pred_lasso = lasso_model_opt.predict(X_test)

# compute mse on test data
y_size = len(y_pred_rr)

error_sum_rr = 0
error_sum_lasso = 0

for i in range(y_size):
    error_sum_rr += (y_test[i] - y_pred_rr[i]) ** 2
    error_sum_lasso += (y_test[i] - y_pred_lasso[i]) ** 2

error_sum_rr = error_sum_rr / y_size
error_sum_lasso = error_sum_lasso / y_size

# print mse on test data
print("MSE from Ridge Regression using optimal lambda_val: ", error_sum_rr)
print("MSE from Lasso using optimal lambda_val: ", error_sum_lasso)
