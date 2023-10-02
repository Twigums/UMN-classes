################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('hw1_q5_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

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

# my guesses for lambda
lambda_vals = [-0.1, -0.01, -0.001, 0]

# same generalization as hw1_q4.py
LDA_lowest_mean = -1
lambda_LDA_opt = 0

for lambda_val in lambda_vals:

    # instantiate LDA object
    LDA_model = MyLDA(lambda_val)

    # call to your CV function to compute error rates for each fold
    LDA_err_vals = my_cross_val(LDA_model, "err_rate", X, y, k = 10)
    LDA_err_mean = np.mean(LDA_err_vals)
    LDA_err_std = np.std(LDA_err_vals)

    # print error rates from CV
    print("Error rate from LDA: ", LDA_err_vals)
    print("Mean error rate from LDA: ", LDA_err_mean)
    print("STD of error rate from LDA: ", LDA_err_std)

    # finds optimal lambda value
    if LDA_lowest_mean == -1:
        LDA_lowest_mean = LDA_err_mean
        lambda_LDA_opt = lambda_val

    elif LDA_lowest_mean > LDA_err_mean:
        LDA_lowest_mean = LDA_err_mean
        lambda_LDA_opt = lambda_val


# instantiate LDA object for best value of lambda
print("Lowest LDA error rate mean at lambda_val = ", lambda_LDA_opt, " with mean = ", LDA_lowest_mean)

LDA_model_opt = MyLDA(lambda_LDA_opt)

# fit model using all training data
LDA_model_opt.fit(X_train, y_train)

# predict on test data
y_class = LDA_model_opt.predict(X_test)

# compute error rate on test data
y_size = len(y_class)

error_vec_LDA = y_class != y_test
error_LDA = sum(error_vec_LDA) / y_size

# print error rate on test data
print("Error rate from LDA using optimal lambda_val: ", error_LDA)

# below is the code for my test plots
"""
X0 = X[y == 0]
X1 = X[y == 1]

x0 = X0[:, 0]
y0 = X0[:, 1]
x1 = X1[:, 0]
y1 = X1[:, 1]

xt0 = []
xt1 = []
yt0 = []
yt1 = []

for i in range(len(y_test)):
    if y_class[i] == 0:
        xt0.append(X_test[i, 0])
        yt0.append(X_test[i, 1])
    if y_class[i] == 1:
        xt1.append(X_test[i, 0])
        yt1.append(X_test[i, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(x0, y0, c="b", label="0")
ax1.scatter(x1, y1, c="r", label="1")
ax1.scatter(xt0, yt0, c="c", label="test0")
ax1.scatter(xt1, yt1, c="m", label="test1")
plt.savefig("plot.png")
"""
