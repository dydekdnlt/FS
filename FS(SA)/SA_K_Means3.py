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

train = pd.read_csv("C:\\Users\\용다윗\\Desktop\\hcvdat0.csv")
# print(train)

train_row = train[['User']]
col_name = ['Age', 'ALB', 'ALP', 'ALT'
    , 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

train_col = train[col_name]

scaler = MinMaxScaler()

scaler.fit_transform(train_col)

A = train_col.to_numpy()

# print(A)

for i in range(0, 615):
    for j in range(11):
        if np.isnan(A[i][j]) == True:
            print(i, j)


def elbow(X):
    sse = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i)
        km.fit(X)
        sse.append(km.inertia_)

    fig = px.line(sse)
    # fig.show()


elbow(A)

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

X = Cluster
new_X = pd.DataFrame(A)

pca = PCA(n_components=2)
pca_tra = pca.fit_transform(X)
# print(pca_tra)
new_pca_tra = pd.DataFrame(pca_tra)
sns.scatterplot(x=X[:, 0], y=X[:, 1], c=km.labels_)

# sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_, data=pca_tra)
# sns.pairplot(new_X)
# sns.pairplot(new_pca_tra)
#plt.show()

k_range = range(2, 30)

best_n = -1
best_silhouette_score = -1
'''
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(A)
    clusters = km.predict(A)

    score = silhouette_score(A, clusters)
    print('k :', k, 'score :', score)

    if score > best_silhouette_score:
        best_n = k
        best_silhouette_score = score

print('best n :', best_n, 'best score :', best_silhouette_score)
'''


def column(matrix, i):
    return [row[i] for row in matrix]


new_col_name = col_name.copy()


def test_def(A, inte, T):
    global B, best_B, last_col_name
    C = []
    km = KMeans(n_clusters=3)
    for i in range(inte):
        B = A.copy()
        B = np.delete(B, i, axis=1)
        km.fit(B)
        clusters = km.predict(B)

        score = silhouette_score(B, clusters)
        C.append(score)
        # print(C)
        T -= 1
    # print(C)

    new_C = sorted(C)
    tmp1 = new_C[0]
    tmp2 = new_C[1]
    print("가장 낮은 점수 : ", tmp1)
    print("두번째로 낮은 점수 : ", tmp2)
    index = C.index(tmp1)
    index2 = C.index(tmp2)
    print("가장 낮은 점수를 갖는 피처 : ", index)
    print("두번째로 낮은 점수를 갖는 피처 : ", index2)
    random_index = []
    random_index.append(index)
    random_index.append(index2)
    final_index = random.choices(random_index, weights=[70, 30])
    B = np.delete(A, final_index, axis=1)
    del new_col_name[final_index[0]]
    print("제거가 된 피처 : ", final_index[0])
    new_T = 10000 - T
    T_list.append(new_T)
    # print(new_T)
    best_score = max(C)
    best_score_index = C.index(best_score)
    if best_Score_List[0] < best_score:
        best_Score_List[0] = best_score
        best_B = B
        last_col_name = new_col_name.copy()
    print("가장 높은 점수 : ", best_score)
    print("현재 피처의 수 : ", len(B[0]))
    print("")
    print("")
    return B, new_T, best_score, best_score_index, best_B, last_col_name


first_score = 0

best_Score_List = []

best_Score_List.append(first_score)

T_list = []
# test_def(X, 11)
T = 10000
while T > 0 or len(B[0]) > 2:
    for i in reversed(range(2, 11)):
        test_def(X, i, T)
        if T > 0:
            X = B
            T -= sum(T_list)
        else:
            break
    break
print("final")
print("")
print("초기 점수 : ", score)
print("가장 높은 점수 : ", best_Score_List[0])

print("초기 피처의 수 : ", len(A[0]))
print("가장 높은 점수의 데이터셋 피처 수 : ", len(best_B[0]))
print("")
km.fit(best_B)
Pd_best_B = pd.DataFrame(best_B)
#sns.scatterplot(x=best_B[:, 0], y=best_B[:, 1], c=km.labels_)
new_pca = PCA(n_components=2)
New_pca_tra = new_pca.fit_transform(best_B)
print(col_name)
print(len(col_name))
print(last_col_name)
print(len(last_col_name))
new_pca_tra2 = pd.DataFrame(New_pca_tra)
sns.scatterplot(x=New_pca_tra[:, 0], y=New_pca_tra[:, 1], c=km.labels_)
# sns.pairplot(new_X)
# sns.pairplot(Pd_best_B)
#plt.show()
