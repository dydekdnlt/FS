import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import pairwise_distances
from scipy.sparse import *


train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = np.array(train[0])
value = np.delete(np.array(train), 0, axis=1)
n_sample, n_feature = value.shape
print(value.shape)
kwargs = {"k": "3"}


def construct_W(X, **kwargs):
    k = kwargs['k']

    k = int(k)

    D = pairwise_distances(X)
    D **= 2
    idx = np.argsort(D, axis=1)

    idx_new = idx[:, 0:k + 1]

    G = np.zeros((n_sample * (k + 1), 3))

    G[:, 0] = np.tile(np.arange(n_sample), (k + 1, 1)).reshape(-1)

    G[:, 1] = np.ravel(idx_new, order='F')

    G[:, 2] = 1

    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_sample, n_sample))

    bigger = np.transpose(W) > W

    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)

    W = W.toarray()
    return W


def get_Laplacian(M):
    S = M.sum(1)

    i_nz = np.where(S > 0)[0]

    S = S[i_nz]

    M = (M[i_nz].T)[i_nz].T

    S = 1 / np.sqrt(S)

    M = S * M

    M = (S * M.T).T

    n = np.size(S)

    M = np.identity(n) - M

    M = (M + M.T) / 2
    return M


new_W = construct_W(value, **kwargs)
print(new_W, new_W.shape)

LS = get_Laplacian(new_W)
print(LS, LS.shape)

testA1 = (value.T @ LS) @ value
print(testA1, testA1.shape)

eigen_value, ul = scipy.linalg.eigh(a=testA1)
print(eigen_value)
print(ul)

Q = np.identity(n=n_feature)
print(Q)