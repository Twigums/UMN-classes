import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta_val):
        self.d = d
        self.max_iters = max_iters
        self.eta_val = eta_val

        # choose random w0 vector
        left_bound = -0.01
        right_bound = 0.01

        # +1 since were adding bias
        self.w = np.random.uniform(left_bound, right_bound, d + 1)
        # derived gradient from 1. but for all w_j in w
        self.alpha = lambda w, x_i: 1 / (1 + np.exp(-w.T.dot(x_i)))
        self.grad_f = lambda w, x_i, y_i: x_i * y_i * (1 - self.alpha(w, x_i)) - x_i * (1 - y_i) * (1 - self.alpha(-w, x_i))

    def fit(self, X, y):
        """
        given update equation is w_{t + 1} = w_t - eta * grad(f(w_t))
        """
        # add bias
        X_rows, X_cols = X.shape
        N = X_rows
        bias = np.ones([N, 1])
        M = np.concatenate((bias, X), axis = 1)

        # given threshold
        threshold = 10 ** -6

        # neglect w0, and save the other w's to find w_bar as discussed in lecture
        w_arr = []

        # we also need to maintain a T for division
        T = 0

        # this portion is for the loss function plot
        f = lambda w, x_i, y_i: y_i * np.log(self.alpha(w, x_i)) + (1 - y_i) * np.log(self.alpha(-w, x_i))
        loss = []

        # looping over max_iters or until the distance meets or is smaller than the given threshold
        for i in range(self.max_iters):
            w = self.w
            T += 1

            # randomly draw data
            index = np.random.randint(0, N)
            x_i = M[index]
            y_i = y[index]

            # this is for the loss function plot
            loss.append(np.abs(f(w, x_i, y_i)))

            # compute subgradient
            g_it = -self.grad_f(w, x_i, y_i)

            w_prev = w
            self.w = w_prev - self.eta_val * g_it
            w_arr.append(self.w)

            # to calculate whether or not we should break out of the loop
            dist = np.linalg.norm(self.w - w_prev)

            # optional printing per 100 iterations
            # if i % 100 == 0:
                # print(i, dist)

            if abs(dist) <= threshold:
                break

        # calculates w_bar as stated in lecture and uses this as w
        w_arr = np.array(w_arr)
        self.w = np.sum(w_arr, axis = 0) / T

        # return losses for the plot
        return loss

    def predict(self, X):
        # add bias
        X_rows, X_cols = X.shape
        N = X_rows
        bias = np.ones([N, 1])
        M = np.concatenate((bias, X), axis = 1)

        y_pred = []

        # the predicted values are alpha(w^T * x_i) for all x_i
        for x_i in M:
            y_val = self.alpha(self.w,x_i)

            # if y_val >= 0.5, then it should have the 1 label; otherwise, 0
            if y_val >= 0.5:
                y_pred.append(1)

            else:
                y_pred.append(0)

        return y_pred


