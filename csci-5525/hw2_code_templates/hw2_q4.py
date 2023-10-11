################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

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

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [1, 10, 100]

# the dimensions, d, must agree for both w and x_i
d = X.shape[1]

# givens
max_iters = 100000
folds = 10

SVM_lowest_mean = -1
eta_opt = 0
c_opt = 0

# SVM
for eta_val in eta_vals:
    for c_val in C_vals:

        # instantiate svm object
        SVM_model = MySVM(d, max_iters, eta_val, c_val)

        # call to your CV function to compute error rates for each fold
        err_vals = my_cross_val(SVM_model, "err_rate", X, y)
        err_mean = np.mean(err_vals)
        err_std = np.std(err_vals)

        # print error rates from CV
        print("Error rate from SVM: ", err_vals)
        print("Average error rate from SVM: ", err_mean)
        print("Standard Deviation of error rate from SVM: ", err_std, "\n")

        if SVM_lowest_mean == -1:
            SVM_lowest_mean = err_mean
            eta_opt = eta_val
            c_opt = c_val

        elif SVM_lowest_mean > err_mean:
            SVM_lowest_mean = err_mean
            eta_opt = eta_val
            c_opt = c_val

# instantiate svm object for best value of eta and C
print("Lowest error rate mean at eta_val = ", eta_opt, "and c_val = ", c_opt, "with mean = ", SVM_lowest_mean)
SVM_model_opt = MySVM(d, max_iters, eta_opt, c_opt)

# fit model using all training data
loss = SVM_model_opt.fit(X_train, y_train)
# SVM_model_opt.fit(X_train, y_train)

# predict on test data
y_pred = SVM_model_opt.predict(X_test)

# compute error rate on test data
test_err = 1 - accuracy_score(y_test, y_pred)

# print error rate on test data
print("Error rate from SVM using optimal eta_val: ", test_err)

# plot details
iterations = len(loss)
x = list(range(iterations))
plt.scatter(x, loss, s = 1)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("SVM-loss.png")
