import numpy as np

class MyPCA():

    def __init__(self, num_reduced_dims):
        self.m = num_reduced_dims
        self.pc_proj = [] # to save principle component direction

    def fit(self, X):

        # find cov matrix of X
        cov = np.cov(X.T)

        # get eigenvalues and eigenvectors
        eigval, eigvec = np.linalg.eig(cov)
        eigval_real = np.real(eigval)
        eigvec_real = np.real(eigvec)

        # we want m PC directions
        for i in range(self.m):

            # choose the max eigenvalue, set it to 0 to prevent it from being chosen again, then save the PC direction vector
            max_idx = np.argmax(eigval_real)
            eigval_real[max_idx] = 0
            pc = eigvec_real[:, max_idx]

            self.pc_proj.append(pc)

        self.pc_proj = np.array(self.pc_proj).T # for cross product size agreement

    def project(self, x):

        # projection is defined as x cross the PC directions
        proj = x @ self.pc_proj

        return proj
