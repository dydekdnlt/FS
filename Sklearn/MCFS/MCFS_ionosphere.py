from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import mutual_info_score
import warnings
import copy
import scipy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn import linear_model
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
from sklearn.impute import SimpleImputer

warnings.filterwarnings(action='ignore')

start = time.time()
train = pd.read_csv("../../DataSet/ionosphere.csv", header=None)
label = np.array(train[34])
value = np.delete(np.array(train), 34, axis=1)
value = np.delete(value, 1, axis=1)

n_sample, n_feature = value.shape

kwargs = {"k": "3"}


def construct_W(X, **kwargs):
    k = kwargs['k']

    k = int(k)

    D = pairwise_distances(X)
    D **= 2
    idx = np.argsort(D, axis=1)
    idx_new = idx[:, 0:k + 1]
    G = np.zeros((n_sample * (k + 1), 3))

    G[:, 0] = np.tile(np.arange(n_sample), (k + 1, 1)).reshape(-1)

    G[:, 1] = np.ravel(idx_new, order='F')

    G[:, 2] = 1

    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_sample, n_sample))

    bigger = np.transpose(W) > W

    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
    print("W 구성")

    return W


def mcfs(X, n_selected_features, **kwargs):
    print(kwargs)

    W = kwargs['W']

    n_cluster = kwargs['n_cluster']

    W = W.toarray()

    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1))) # 대각 행렬 D
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]

    eigen_value, ul = scipy.linalg.eigh(a=W)
    print("eigen_value : ", eigen_value)
    print("ul : ", ul)
    print(ul.shape)
    Y = np.dot(W_norm, ul[:, -1 * n_cluster - 1:-1])
    print(Y.shape)

    n_sample, n_feature = X.shape
    W = np.zeros((n_feature, n_cluster))

    for i in range(n_cluster):
        clf = linear_model.Lars(normalize=False, n_nonzero_coefs=n_selected_features)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
        print(clf.coef_)

    return W


def feature_ranking(W):
    mcfs_score = W.max(1)
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]
    print("idx : ", idx)
    return idx


X = construct_W(value, **kwargs)

num_fea = 15
num_cluster = 2

Weight = mcfs(value, n_selected_features=num_fea, W=X, n_cluster=num_cluster)

idx = feature_ranking(Weight)

selected_features = value[:, idx[0:num_fea]]
print("MCFS")
print(selected_features)
new_selected_features = pd.DataFrame(selected_features)
new_mat_file_value = pd.DataFrame(value)
print(new_mat_file_value)
print(new_selected_features)
# 고유 문제 해결 후 > LARs 알고리즘 사용 > MCFS 점수 순 나열

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_selected_features, label,
                                                                    test_size=0.25)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print(minScore)

end = time.time()
print(f"{end - start: .2f} sec")