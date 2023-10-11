import numpy as np

class MySVM:

    def __init__(self, d, max_iters, eta_val, c):
        self.d = d
        self.max_iters = max_iters
        self.eta_val = eta_val
        self.c = c

        # choose random w0 vector
        left_bound = -0.01
        right_bound = 0.01

        # +1 since were adding bias
        self.w = np.random.uniform(left_bound, right_bound, d + 1)

    def fit(self, X, y):
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
        f = lambda w, x_i, y_i: max(0, 1 - y_i * (w.dot(x_i)))
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
            if 1 - y_i * (w.dot(x_i)) > 0:
                loss.append(f(w, x_i, y_i))

            # compute subgradient; I am going to assume we just want grad_f = w + c * conditional, where the conditional is stated from 3.
            # I don't think we're summing since we're just choosing a random point
            if y_i * (w.dot(x_i)) >= 1:
                hinge = 0

            else:
                hinge = -y_i * x_i

            grad_f = w + self.c * hinge

            w_prev = w
            self.w = w_prev - self.eta_val * grad_f
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

        # the predicted values are just the data (M) * w signed
        y_pred = np.sign(M.dot(self.w.T))

        return y_pred
