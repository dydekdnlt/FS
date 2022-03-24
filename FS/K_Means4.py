import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
import random
train = pd.read_csv("C:\\Users\\용다윗\\Desktop\\Sales_Transactions_Dataset_Weekly.csv")
#print(train)

train_row = train[['Product_Code']]
col_name = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8',
                   'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16',
                   'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24',
                   'W25', 'W26', 'W27', 'W28', 'W29', 'W30', 'W31', 'W32',
                   'W33', 'W34', 'W35', 'W36', 'W37', 'W38', 'W39', 'W40',
                   'W41', 'W42', 'W43', 'W44', 'W45', 'W46', 'W47', 'W48',
                   'W49', 'W50', 'W51']
train_col = train[col_name]

scaler = MinMaxScaler()

scaler.fit_transform(train_col)

A = train_col.to_numpy()

km = KMeans(n_clusters=2)
km.fit(A)
centers = km.cluster_centers_
#print(km.labels_)
#print(km.cluster_centers_)
Cluster = A.copy()
clusters = km.predict(A)
score = silhouette_score(A, clusters)
new_km_labels = pd.DataFrame(km.labels_)
#print(new_km_labels)
print(score)

X = Cluster
new_X = pd.DataFrame(A)

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
#print(pca_tra)
new_pca_tra = pd.DataFrame(pca_tra)

#sns.scatterplot(x=X[:,0], y=X[:, 1], c=km.labels_, data=X)

sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_, data=pca_tra)
#sns.pairplot(new_X)
#plt.show()
