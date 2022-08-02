from sklearn.datasets import load_iris
import pandas as pd
from scipy import linalg
import numpy as np

iris = load_iris()
train = pd.read_csv("../DataSet/wine.csv")
#print(train)
#print(train.shape)
X = np.delete(np.array(train), 0, axis=1)

#print(iris)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df1 = df.head()

np_df1 = df.to_numpy()

u = np_df1 @ np_df1.T
print(u)
eig = np.linalg.eig
value = eig(u)[0]
vector = eig(u)[1]
print(value)
print(vector)

Vt = np_df1.T @ np_df1
value2 = eig(Vt)[0]
print(value2)
vector2 = eig(Vt)[1]
print("vector2")
print(vector2.T)
U, s, Vh = linalg.svd(np_df1)
print(s)
print(Vh)
for k in range(len(value)):
    print(value[k]**1/2)

for j in range(len(value2)):
    print(value2[j]**1/2)

for i in range(len(s)):
    s[i] = round(s[i], 2)
    print(i, s[i])

for i in range(4):
    test_list = []
    for j in range(150):
        test_list.append(np_df1[j][i])
    print(test_list)
    print(np.mean(test_list))
    print(np.std(test_list))
