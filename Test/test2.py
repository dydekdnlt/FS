import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy.linalg
import skfuzzy as fuzz
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error

movies = pd.read_csv('C:/Users/ForYou/Desktop/ml-1m/movies.csv')
ratings = pd.read_csv('C:/Users/ForYou/Desktop/ml-1m/ratings.csv')

rating_movies = pd.merge(ratings, movies, on='MovieID')

ratings_matrix = rating_movies.pivot_table('rating', index='UserID', columns='title')
ratings_matrix.fillna(0, inplace=True)
ratings_matrix_T = ratings_matrix.transpose()

cluster = 10
kmeans = KMeans(n_clusters=cluster, random_state=0)
X = kmeans.fit_predict(ratings_matrix_T)

print(kmeans.labels_)
cntr, u, u0, d, _, _, _ = fuzz.cluster.cmeans(ratings_matrix, cluster, 2, error=0.005, maxiter=1000, init=None)
print("train", ratings_matrix_T.shape, sep='\n')
print("Fuzzy 클러스터 별 속할 가능성", u, sep='\n')
print("K-means 클러스터 레이블", X, sep='\n')

print(type(ratings_matrix_T))
U, sigma, V = svds(ratings_matrix_T.to_numpy(), k=cluster)
# U, sigma, V = np.linalg.svd(ratings_matrix_T)
a, b = U.shape
np_sigma = np.zeros((b, b))

print("U", U.shape, sep='\n')
print("sigma", sigma.shape, sep='\n')
print(sigma)
print("V", V.shape, sep='\n')

for i in range(len(sigma)):
    np_sigma[i][i] = sigma[i]

print("np_sigma", np_sigma.shape)

test_svd = U @ np_sigma
print("test_svd", test_svd.shape)
cntr2, u2, _, _, _, _, _ = fuzz.cluster.cmeans(test_svd.T, cluster, 2, error=0.005, maxiter=1000, init=None)
print("u2", u2)
print("원본 행렬의 k-means 클러스터 레이블", X, sep='\n')

kmeans_test_svd = kmeans.fit_predict(test_svd)

nmf = NMF(n_components=cluster)
W = nmf.fit_transform(ratings_matrix_T)
print("W 확인", W.shape)

print("Truncated SVD 행렬의 k-means 클러스터 레이블", kmeans_test_svd, sep='\n')
pca = PCA(n_components=2)
pca_tra = pca.fit_transform(ratings_matrix_T)
pca_tra2 = pca.fit_transform(test_svd)
pca_tra3 = pca.fit_transform(W)

#plt.bar(x, score_list[0])
#plt.xticks(x, column_name)
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9, 4), ncols=3)
ax1.scatter(x=pca_tra[:, 0], y=pca_tra[:, 1], c=X)
ax2.scatter(x=pca_tra2[:, 0], y=pca_tra2[:, 1], c=X)
ax3.scatter(x=pca_tra3[:, 0], y=pca_tra3[:, 1], c=X)
ax1.set_title('원본 행렬의 k-means')
ax2.set_title("Truncated SVD 행렬의 k-means")
ax3.set_title("NMF")
# sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=kmeans.labels_)
plt.show()
# 열 기준 X, 행 기준으로 다시 구성
