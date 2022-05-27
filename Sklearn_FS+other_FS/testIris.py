import pandas as pd
import numpy as np
import scipy
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
train = pd.read_csv('C:\\Users\\용다윗\\PycharmProjects\\pythonProject7\\iris_data.csv', index_col=0)

new_train = train.loc[[0, 1, 50, 51, 100, 101]]

print(new_train)

A = kneighbors_graph(new_train, 3)

B = pairwise_distances(new_train)
print(B)
for i in range(6):
    for j in range(6):
        B[i][j] = round(B[i][j], 2)
print(B)
zero_matrix = np.zeros([6, 6])
A = A.toarray()
print(A)
t = 10

for i in range(6):
    for j in range(6):
        if A[i][j] == 1:
            zero_matrix[i][j] = np.exp(-((B[i][j]**2)/t))

print(zero_matrix)

one = np.ones(6)

D = np.diag(zero_matrix @ one.T)
print(D)

L = D - zero_matrix

print(L)
X = new_train.to_numpy()
I = np.identity(n=4)


test = (X.T @ L @ X) + I
'''
for i in range(4):
    for j in range(4):
        test[i][j] = round(test[i][j], 2)
'''
print(test)
eigen_value, ul = scipy.linalg.eigh(a=test)

print(I)
print("고유값 : ", eigen_value)
print("고유행렬 : ", ul)