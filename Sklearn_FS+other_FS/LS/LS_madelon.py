from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, accuracy_score
import time

start = time.time()
train = pd.read_csv("../../DataSet/madelon.csv", header=0, index_col=0)
label = np.array(train['500'])
value = np.delete(np.array(train), label, axis=1)
print(pd.DataFrame(value))
n_sample, n_feature = value.shape
# print(n_sample, n_feature)
A = kneighbors_graph(value, 6)
print(A)
B = pairwise_distances(value)
print(B)
zero_matrix = np.zeros([n_sample, n_sample])
A = A.toarray()

select_f = 40

valueT = value.T

t = 1000
for i in range(n_sample):
    for j in range(n_sample):
        if A[i][j] == 1:
            zero_matrix[i][j] = np.exp(-((B[i][j]**2)/t))
            # print(np.exp(-((B[i][j]**2)/t)))


one = np.ones(n_sample)

D = np.diag(zero_matrix @ one.T)

L = D - zero_matrix

pd_L = pd.DataFrame(L)


testA = valueT[0] - (valueT[0].T @ D @ one / (one.T @ D @ one)) * one

testlist = []
for i in range(n_feature):
    testlist.append(valueT[i] - (valueT[i].T @ D @ one / (one.T @ D @ one)) * one)

LS = pd.DataFrame(testlist)
LS = LS.to_numpy()


score_list = []

for i in range(n_feature):
    score_list.append(LS[i].T @ L @ LS[i] / LS[i].T @ D @ LS[i])
print(score_list)
for j in range(len(score_list)):
    if np.isnan(score_list[j]):
        print("nan", j)
        score_list[j] = 0
AA = sorted(score_list)
index_list = []
for i in range(select_f):
    index_list.append(score_list.index(AA[i]))
print(index_list)
selected_features = value[:, index_list[0:]]
print(selected_features)

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(selected_features, label, test_size=0.50)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print("LS 스코어 : ", minScore)
end = time.time()
print(f"{end - start: .5f} sec")