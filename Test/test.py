import numpy as np
import matplotlib.pylab as plt
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from timeit import default_timer as timer
from datetime import timedelta
start = timer()
mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)

print(type(mat_file))

for i in mat_file:
    print(i)

mat_file_value = mat_file["fea"]
print(len(mat_file_value), len(mat_file_value[0]))
print(mat_file_value)

scaler = MinMaxScaler()

scaler.fit_transform(mat_file_value)
'''
k_range = range(2, 40)

best_n = -1
best_silhouette_score = -1

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(mat_file_value)
    clusters = km.predict(mat_file_value)

    score = silhouette_score(mat_file_value, clusters)
    print('k :', k, 'score :', score)

    if score > best_silhouette_score:
        best_n = k
        best_silhouette_score = score

print('best n :', best_n, 'best score :', best_silhouette_score)
'''

# A = mat_file_value.to_numpy()
km = KMeans(n_clusters=6)
km.fit(mat_file_value)
centers = km.cluster_centers_
# print(km.labels_)
# print(km.cluster_centers_)

X = mat_file_value.copy()
clusters = km.predict(mat_file_value)
score = silhouette_score(mat_file_value, clusters)
new_km_labels = pd.DataFrame(km.labels_)
# print(new_km_labels)
print(score)

end = timer()
print(timedelta(seconds=end - start))

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
# print(pca_tra)
new_pca_tra = pd.DataFrame(pca_tra)
new_X = pd.DataFrame(X)
#sns.scatterplot(x=X[:, 0], y=X[:, 1], c=km.labels_)
sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_)
# sns.pairplot(new_X)
plt.show()
