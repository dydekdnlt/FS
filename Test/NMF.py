import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
import skfuzzy as fuzz
from scipy.sparse.linalg import svds
import seaborn as sns

df = pd.read_csv('C:\\Users\\ForYou\\Desktop\\DataSet\\letter.csv', header=None)

print(df)

label = np.array(df[0])

df = df.drop([0], axis=1)

print(df)

cluster = 5
kmeans = KMeans(n_clusters=cluster, random_state=0)

nmf = NMF(n_components=cluster)
W = nmf.fit_transform(df)
print("W 확인", W.shape)

kmeans_test_nmf = kmeans.fit_predict(W)

print(kmeans.labels_)

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(W, cluster, 2, error=0.005, maxiter=1000, init=None)

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(df)

sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1], c=kmeans_test_nmf)
plt.show()