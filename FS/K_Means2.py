import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score

train = pd.read_csv("C:\\Users\\용다윗\\Desktop\\perfume_data.csv")
# print(train)
train_row = train[['name']]
col_name = ['1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28']
train_col = train[col_name]

scaler = MinMaxScaler()

scaler.fit_transform(train_col)

A = train_col.to_numpy()

km = KMeans(n_clusters=3)
km.fit(A)
centers = km.cluster_centers_
# print(km.labels_)
# print(km.cluster_centers_)
Cluster = A.copy()
clusters = km.predict(A)
score = silhouette_score(A, clusters)
new_km_labels = pd.DataFrame(km.labels_)
# print(new_km_labels)
print(score)

X = Cluster
pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
# print(pca_tra)
new_X = pd.DataFrame(X)
new_pca_tra = pd.DataFrame(pca_tra)
#sns.scatterplot(x=X[:, 0], y=X[:, 1], c=km.labels_)
sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1], c=km.labels_)
# sns.pairplot(new_X)
plt.show()
