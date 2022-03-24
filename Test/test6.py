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
import scipy.io
mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)

print(type(mat_file))

for i in mat_file:
    print(i)

mat_file_value = mat_file["fea"]
print(len(mat_file_value), len(mat_file_value[0]))
print(mat_file_value)

col_name = mat_file_value[0]
print(col_name)
new_mat_file_value = pd.DataFrame(mat_file_value)
print(new_mat_file_value)
std_list = []
sort_std_list = []
print(new_mat_file_value[0])
print(len(new_mat_file_value[0]))
for i in range(4096):
    std_list.append(np.std(new_mat_file_value[i]))
print("std 리스트 :", std_list)
score_list = []

'''
for i in range(1, 1024):  # 피처 수 1일때 사용

    final_list = [i]

    new_train_col = new_mat_file_value.iloc[:, final_list]

    # print(new_train_col)

    scaler = MinMaxScaler()

    scaler.fit_transform(new_train_col)

    A = new_train_col.to_numpy()

    # print(A)

    km = KMeans(n_clusters=3)
    km.fit(A)
    centers = km.cluster_centers_
    # print(km.labels_)
    # print(km.cluster_centers_)
    Cluster = A.copy()
    clusters = km.predict(A)
    score = silhouette_score(A, clusters)
    new_km_labels = pd.DataFrame(km.labels_)
    print("피처의 수 :", i, "-", score)
    score_list.append(score)
    final_list.remove(i)

print("최소값 :", min(score_list))
print("최소값 인덱스 :", score_list.index(min(score_list)))
print("최대값 :", max(score_list))
print("최대값 인덱스:", score_list.index(max(score_list)))

high_final_list = [31]
km = KMeans(n_clusters=4)
new_train_col = new_mat_file_value.iloc[:, high_final_list]
km.fit(new_train_col)
print(new_train_col)
sns.scatterplot(x=new_mat_file_value.iloc[:, 0], y=new_train_col.iloc[:, 0], c=km.labels_)
plt.xlabel('')
plt.show()
'''


for i in range(4096):
    std_list.append(np.std(new_mat_file_value[i]))

sort_std_list.append(sorted(std_list))

#print("표준 편차 : ", std_list)

new_B = sort_std_list[0]
new_B.sort(reverse=True)
print("내림차순 정렬된 표준 편차 : ", new_B)
print()
print()
high_count = 0
high_score = 0
first_score = 0
high_score_list = []
high_final_list = []
count = 1
while count < 30:
    final_list = []
    for i in range(count):

        final_list.append(std_list.index(new_B[i]))

    new_train_col = new_mat_file_value.iloc[:, final_list]

    #print(new_train_col)

    scaler = MinMaxScaler()

    scaler.fit_transform(new_train_col)

    A = new_train_col.to_numpy()

    #print(A)

    km = KMeans(n_clusters=6)
    km.fit(A)
    centers = km.cluster_centers_
    #print(km.labels_)
    #print(km.cluster_centers_)
    Cluster = A.copy()
    clusters = km.predict(A)
    score = silhouette_score(A, clusters)
    new_km_labels = pd.DataFrame(km.labels_)
    #print(new_km_labels)
    print("피처의 수 :", count, "-", score)
    if score > high_score:
        high_final_list = final_list
        high_score = score
        high_score_list = A
        high_count = count

    count += 1


print("초기 데이터셋 점수 :", first_score) # 초기 데이터셋 점수
print("가장 높은 점수 :", high_score) # 가장 높은 점수
print("선택된 피처 인덱스 :", high_final_list) # 선택된 피처 인덱스
#print(high_score_list) # 가장 높은 점수의 데이터셋
#print("점수 증가율 :", (high_score - first_score) / first_score * 100)
if len(high_final_list) == 1:
    km = KMeans(n_clusters=4)
    new_train_col = new_mat_file_value.iloc[:, high_final_list]
    km.fit(new_train_col)
    print(new_train_col)
    sns.scatterplot(x=new_mat_file_value.iloc[:, 0], y=new_train_col.iloc[:, 0], c=km.labels_)
    plt.xlabel('')
    plt.show()
else:
    km = KMeans(n_clusters=4)
    km.fit(high_score_list)
    centers = km.cluster_centers_
    #print(km.labels_)
    #print(km.cluster_centers_)
    Cluster = high_score_list.copy()
    X = Cluster

    pca = PCA(n_components=2)
    pca_tra = pca.fit_transform(X)
    #print(pca_tra)
    #new_pca_tra = pd.DataFrame(pca_tra)

    #sns.scatterplot(x=X[:,0], y=X[:, 1], c=km.labels_)

    sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_, data=pca_tra)
    #sns.pairplot(new_X)
    plt.show()
