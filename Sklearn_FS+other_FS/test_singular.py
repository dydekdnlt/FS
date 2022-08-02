from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from scipy import linalg
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import seaborn as sns

iris = load_iris()
train = pd.read_csv("../DataSet/wine.csv")
#print(train)
#print(train.shape)
X = np.delete(np.array(train), 0, axis=1)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df1 = df.head()

np_df1 = df1.to_numpy()

print(np_df1)

A = np_df1

U = A @ A.T
V = A.T @ A
print(U)

print(V)


eig = np.linalg.eig

U_value = eig(U)[0]
U_vector = eig(U)[1]

V_value = eig(V)[0]

V_vector = eig(V)[1]

print("U_value", U_value, sep='\n')
print("U_vector", U_vector, sep='\n')
print("V_value", V_value, sep='\n')
print("V_vector", V_vector, sep='\n')
V_value = V_value ** (1/2)
sigma = np.zeros((5, 4))
for i in range(4):
    sigma[i][i] = V_value[i]
print("Sigma", sigma, sep='\n')

'''
norms1 = np.linalg.norm(U_vector, axis=1)
norms2 = np.linalg.norm(V_vector, axis=1)

U_vector = U_vector / norms1
V_vector = V_vector / norms2
'''
print("M", U_vector @ sigma @ V_vector.T, sep='\n')
appA = np.dot(U_vector, np.dot(sigma, V_vector.T))
print("기존 행렬과의 차이", np_df1 - appA, sep='\n')

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
# print(pca_tra)
new_pca_tra = pd.DataFrame(pca_tra)
new_X = pd.DataFrame(X)
print(new_X)
#sns.scatterplot(x=X[:, 0], y=X[:, 1])
# sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1])
#sns.pairplot(new_X)
plt.show()
