import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import pairwise_distances
from scipy.sparse import *
from scipy.linalg import solve_triangular
from sklearn.decomposition import NMF
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = np.array(train[0])
X = np.delete(np.array(train), 0, axis=1)
n, d = X.shape

# 파라미터 설정 부분
m = 5

lambda1 = 1
lambda2 = 1
ITER1 = 20

clustering = KMeans(n_clusters=m).fit_predict(X)
First_F = np.zeros([n, m])
for i in range(n):
    First_F[i][clustering[i]] = 1
T = First_F

nmf1 = NMF(n_components=m, init='nndsvd')
nmf2 = NMF(n_components=m, init='nndsvd')
A = nmf1.fit_transform(X)
W = nmf1.components_
print("X", X.shape)
print("W", W.shape)
E = nmf2.fit_transform(T)

print("E", E.shape)

B = nmf2.components_

print("B", B.shape)
print("T", T.shape)

d2 = np.ones(d)

X = X - np.mean(X, axis=1) @ np.ones(n)
b = np.zeros((m, 1))
print("b", b.shape)
XX = X.T @ X

test_n = np.ones((n, 1))
print("test_n", test_n.shape)
H = np.random.randint(0, 1, size=E.shape)
print("H", H.shape)
WXB = ((W @ X.T) + (b @ test_n.T)).T @ B.T
for i in range(ITER1):
    ITER2 = 1

    while ITER2 <= 20:
        test_svd = WXB + (lambda2 * H)
        U, s, V = np.linalg.svd(test_svd)
        print("U, V", U.shape, V.shape)
        E = V @ np.eye(m, n) @ U.T
        H = 0.5 * (E + abs(E))
        H = H.T
        ITER2 += 1
    print("E", E.shape)
    AB = E @ ((W @ X.T) + b @ np.ones((1, n))).T
    print("AB", AB.shape)
    LE, sigma, RE = np.linalg.svd(AB.T)
    print("LE, RE", LE.shape, RE.shape)
    new_B = RE @ np.eye(m, m) @ LE.T
    B = new_B
    print("B", B.shape)

    EB = E.T @ B
    print("EB", EB.shape)
    D2 = np.diag(d2)
    print("D2", D2.shape)
    new_b = np.mean(EB.T - W @ X.T, axis=1)

    pd_new_b = pd.DataFrame(new_b)

    np_new_b = np.matrix(pd_new_b)

    b = np_new_b
    inv = np.linalg.inv(XX + (lambda1 * D2))

    W = (np.linalg.inv(XX + (lambda1 * D2)) @ X.T) @ (EB - np.ones((n, 1)) @ b.T)
    print(W)
    for j in range(d):
        d2[j] = 1 / (2 * np.sqrt(np.sum(W[j])))
        if np.isnan(d2[j]):
            d2[j] = 0
    print(d2)


