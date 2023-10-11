################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MyLogisticRegression import MyLogisticRegression

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

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
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val

# to be consistent with everything, i will use accuracy_score that is used under my_cross_val.py for calculating error rate
from sklearn.metrics import accuracy_score

# for plotting purposes
import matplotlib.pyplot as plt

eta_vals = [0.001, 0.01, 0.1]

# the dimensions, d, must agree for both w and x_i
d = X.shape[1]

# givens
max_iters = 100000
folds = 10

LR_lowest_mean = -1
eta_opt = 0

# Logistic Regression
for eta_val in eta_vals:

    # instantiate logistic regression object
    LR_model = MyLogisticRegression(d, max_iters, eta_val)

    # call to your CV function to compute error rates for each fold
    err_vals = my_cross_val(LR_model, "err_rate", X, y)
    err_mean = np.mean(err_vals)
    err_std = np.std(err_vals)

    # print error rates from CV
    print("Error rate from Logistic Regression: ", err_vals)
    print("Average error rate from Logistic Regression: ", err_mean)
    print("Standard Deviation of error rate from Logistic Regression: ", err_std, "\n")

    # chooses best eta based on respective means
    if LR_lowest_mean == -1:
        LR_lowest_mean = err_mean
        eta_opt = eta_val

    elif LR_lowest_mean > err_mean:
        LR_lowest_mean = err_mean
        eta_opt = eta_val

# instantiate logistic regression object for best value of eta
print("Lowest error rate mean at eta_val = ", eta_opt, "with mean = ", LR_lowest_mean)
LR_model_opt = MyLogisticRegression(d, max_iters, eta_opt)

# fit model using all training data

loss = LR_model_opt.fit(X_train, y_train)

# predict on test data
y_pred = LR_model_opt.predict(X_test)

# compute error rate on test data
test_err = 1 - accuracy_score(y_test, y_pred)

# print error rate on test data
print("Error rate from Logistic Regression using optimal eta_val: ", test_err)

# plot details
iterations = len(loss)
x = list(range(iterations))
plt.scatter(x, loss, s = 1)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("LR-loss.png")
