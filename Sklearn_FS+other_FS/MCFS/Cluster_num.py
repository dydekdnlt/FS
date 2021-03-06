from sklearn.cluster import KMeans
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
mat_file_name = "../../DataSet/ORL_32x32.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"] # 패턴
mat_file_label = mat_file["gnd"] # 레이블
value = np.array(mat_file_value)
label = np.array(mat_file_label)

'''
train = pd.read_csv("../../DataSet/madelon.csv", header=0, index_col=0)
label = np.array(train['500'])
value = np.delete(np.array(train), label, axis=1)
'''
inertia_arr = []
k_range = range(2, 40)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=200)
    model.fit(value)
    inertia = model.inertia_
    inertia_arr.append(inertia)
    print('k : ', k, 'inertia : ', inertia)

inertia_arr = np.array(inertia_arr)
print(inertia_arr)

plt.plot(k_range, inertia_arr, marker='o')
plt.xlabel('Cluster Number')
plt.ylabel('inertia')
plt.show()

'''

Yale_64x64 : 10
YaleB_32x32 : 4
ORL_32x32 : 20
Wine : 7
Hepatitis : 21
Ionosphere : 19
SpamBase : 14
Arrhythmia : 13
Madelon : 15

'''