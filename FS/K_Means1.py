import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score


train = pd.read_csv("../DataSet/google_review_ratings.csv", dtype=float)
# print(train)

train_row = train[['User']]

col_name = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5',
            'Category 6', 'Category 7', 'Category 8', 'Category 9', 'Category 10',
            'Category 11', 'Category 12', 'Category 13', 'Category 14', 'Category 15',
            'Category 16', 'Category 17', 'Category 18', 'Category 19', 'Category 20',
            'Category 21', 'Category 22', 'Category 23', 'Category 24']

train_col = train[col_name]

scaler = MinMaxScaler()

scaler.fit_transform(train_col)

A = train_col.to_numpy()
km = KMeans(n_clusters=37)
km.fit(A)
centers = km.cluster_centers_
# print(km.labels_)
# print(km.cluster_centers_)

X = A.copy()
clusters = km.predict(A)
score = silhouette_score(A, clusters)
new_km_labels = pd.DataFrame(km.labels_)
# print(new_km_labels)
print(score)

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
# print(pca_tra)
new_pca_tra = pd.DataFrame(pca_tra)
new_X = pd.DataFrame(X)
#sns.scatterplot(x=X[:, 0], y=X[:, 1], c=km.labels_)
#sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1], c=km.labels_)
sns.pairplot(new_X)
plt.show()