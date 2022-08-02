from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, accuracy_score
import time
from sklearn.cluster import SpectralClustering, KMeans

start = time.time()
train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = np.array(train[0])
X = np.delete(np.array(train), 0, axis=1)
n_sample, n_feature = X.shape

cluster = 3
select_f = 6
p = 10 # 가중치 행렬 W 구성에 필요한 파라미터
alpha, beta, gamma = 10**-6, 10**-6, 10**8

knn_graph = kneighbors_graph(X, cluster)
distance = pairwise_distances(X)

clustering = SpectralClustering(n_clusters=cluster, assign_labels='discretize').fit_predict(X)

First_F = np.zeros([n_sample, cluster])
for i in range(n_sample):
    First_F[i][clustering[i]] = 1
F = First_F
print(F, F.shape)
W = np.zeros([n_sample, n_sample])
knn_graph = knn_graph.toarray()

DI = np.identity(n_sample)

for i in range(n_sample):
    for j in range(n_sample):
        if knn_graph[i][j] == 1:
            W[i][j] = np.exp(-((distance[i][j]**2)/p))

one = np.ones(n_sample)

Diag = np.diag(W @ one.T)

L = Diag - W

pd_L = pd.DataFrame(L)

I = np.identity(n_sample)
D = np.identity(n_feature)
M = L + alpha * (I - X @ np.linalg.inv((X.T @ X) + (beta * D)) @ X.T)

for i in range(n_sample):
    for j in range(cluster):
        F[i][j] = gamma * F[i][j] / (M @ F + gamma * (F @ F.T @ F))[i][j]

W = np.linalg.inv(X.T @ X + beta * D) @ X.T @ F

for i in range(n_feature):
    a = 1 / 2 * np.linalg.norm(W[i])
    D[i][i] = a

for i in range(30):
    M = L + alpha * (I - X @ np.linalg.inv((X.T @ X) + (beta * D)) @ X.T)
    test_F = (gamma * F) / (M @ F + gamma * (F @ F.T @ F))
    for j in range(n_sample):
        for k in range(cluster):
            F[j][k] = F[j][k] * test_F[j][k]
    W = np.linalg.inv(X.T @ X + beta * D) @ X.T @ F
    for l in range(n_feature):
        a = 1 / 2 * np.linalg.norm(W[l])
        D[l][l] = a

W_Distance = []
for i in range(n_feature):
    W_Distance.append(D[i][i])
print(W_Distance)
Sort_W_Distance = sorted(W_Distance, reverse=True)
print(Sort_W_Distance)
selected_feature = []
for i in range(select_f):
    selected_feature.append(W_Distance.index(Sort_W_Distance[i]))
print(selected_feature)

X = pd.DataFrame(X)
print(X)
final_X = X.loc[:, sorted(selected_feature)]

print(final_X)


new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(final_X, label, test_size=0.50)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print("NDFS 스코어 : ", minScore)
end = time.time()
print(f"{end - start: .5f} sec")
