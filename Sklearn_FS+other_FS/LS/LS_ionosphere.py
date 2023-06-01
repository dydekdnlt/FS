from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, accuracy_score
import time

start = time.time()
train = pd.read_csv("../../DataSet/ionosphere.csv", header=None)
label = np.array(train[34])
value = np.delete(np.array(train), 34, axis=1)
value = np.delete(value, 1, axis=1)
n_sample, n_feature = value.shape
A = kneighbors_graph(value, 3)
B = pairwise_distances(value)
zero_matrix = np.zeros([n_sample,n_sample])
A = A.toarray()
select_f = 15
print(A.shape)
print(value)
valueT = value.T
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
for i in range(select_f):
    index_list.append(score_list.index(AA[i]))
print(index_list)
selected_features = value[:, index_list[0:]]
print(selected_features)

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(selected_features, label, test_size=0.20)
#new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
#print("new score : ", new_scores)
#print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = accuracy_score(new_Y_test.ravel(), new_Y_pred)
print("LS 스코어 : ", minScore)
end = time.time()
print(f"{end - start: .5f} sec")