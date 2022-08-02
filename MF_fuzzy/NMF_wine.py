import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
import skfuzzy as fuzz
from scipy.sparse.linalg import svds
import seaborn as sns
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
import time

start = time.time()

df = pd.read_csv('C:\\Users\\ForYou\\Desktop\\DataSet\\wine.csv', header=None)
label = np.array(df[0])
print(df)
print(label)
df = df.drop([0], axis=1)
df_numpy = df.to_numpy()
n_sample, n_feature = df_numpy.shape

A = kneighbors_graph(df_numpy, 3)
B = pairwise_distances(df_numpy)
zero_matrix = np.zeros([n_sample, n_sample])
A = A.toarray()
select_f = 9

valueT = df_numpy.T
print(valueT[0])
t = 10
for i in range(n_sample):
    for j in range(n_sample):
        if A[i][j] == 1:
            zero_matrix[i][j] = np.exp(-((B[i][j]**2)/t))
            print(zero_matrix[i][j])

one = np.ones(n_sample)
print(one)
D = np.diag(zero_matrix @ one.T)
print(zero_matrix)
print(D)
L = D - zero_matrix
print(L)
pd_L = pd.DataFrame(L)
print(pd_L)

testA = valueT[0] - (valueT[0].T @ D @ one / (one.T @ D @ one)) * one
print(testA)
testlist = []
for i in range(n_feature):
    testlist.append(valueT[i] - (valueT[i].T @ D @ one / (one.T @ D @ one)) * one)

LS = pd.DataFrame(testlist)
LS = LS.to_numpy()
print(LS.T.shape)

score_list = []

for i in range(n_feature):
    score_list.append(LS[i].T @ L @ LS[i] / LS[i].T @ D @ LS[i])
print(score_list)
AA = sorted(score_list)
index_list = []
print(AA, len(AA))
for i in range(select_f):
    index_list.append(score_list.index(AA[i]))
print(index_list)
selected_features = df_numpy[:, index_list[0:]]
print(selected_features)
cluster = 3
kmeans = KMeans(n_clusters=cluster, random_state=0)

nmf = NMF(n_components=cluster)
W = nmf.fit_transform(selected_features)
H = nmf.components_
print("W 확인", W, W.shape, sep='\n')
print("H 확인", H, H.shape, sep='\n')

test_nmf = W
kmeans_test_nmf = kmeans.fit_predict(test_nmf)

print(kmeans.labels_)

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(test_nmf.T, cluster, 2, error=0.005, maxiter=1000, init=None)
defuzz = []

for i in range(len(u[0])):
    a = []
    for j in range(len(u)):
        a.append(u[j][i])
    defuzz.append(a.index(max(a)))


print(defuzz)


c1 = list(label).count(1)
c2 = list(label).count(2)
c3 = list(label).count(3)
print(c1, c2, c3)
count_0, count_1, count_2 = 0, 0, 0
for i in range(0, c1):

    if defuzz[i] == 0:
        count_0 += 1
    elif defuzz[i] == 1:
        count_1 += 1
    elif defuzz[i] == 2:
        count_2 += 1
print(max(count_0, count_1, count_2), count_0, count_1, count_2)
count_0, count_1, count_2 = 0, 0, 0
for i in range(c1, c1 + c2):

    if defuzz[i] == 0:
        count_0 += 1
    elif defuzz[i] == 1:
        count_1 += 1
    elif defuzz[i] == 2:
        count_2 += 1
print(max(count_0, count_1, count_2), count_0, count_1, count_2)
count_0, count_1, count_2 = 0, 0, 0
for i in range(c1 + c2, len(label)):
    if defuzz[i] == 0:
        count_0 += 1
    elif defuzz[i] == 1:
        count_1 += 1
    elif defuzz[i] == 2:
        count_2 += 1
print(max(count_0, count_1, count_2), count_0, count_1, count_2)

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(test_nmf, label, test_size=0.20)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=5)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = accuracy_score(new_Y_test.ravel(), new_Y_pred)
print("정확도(accuracy) : ", minScore)
precision = precision_score(new_Y_test.ravel(), new_Y_pred, average='macro')
print("정밀도(precision) : ", precision)
recall = recall_score(new_Y_test.ravel(), new_Y_pred, average='macro')
print("민감도(recall) : ", recall)

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(test_nmf)
print("time : ", time.time() - start)
sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1], c=defuzz)
plt.show()