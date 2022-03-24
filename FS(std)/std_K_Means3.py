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
train = pd.read_csv("C:\\Users\\용다윗\\Desktop\\hcvdat0.csv")
# print(train)

train_row = train[['User']]
col_name = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
train_col = train[col_name]
new_col_name = col_name
std_list = []
sort_std_list = []

for i in col_name:
    std_list.append(np.std(train_col[i]))

sort_std_list.append(sorted(std_list))

print("표준 편차 : ", std_list)
#print(sort_std_list)
new_B = sort_std_list[0]

new_B.sort(reverse=True)
print("정렬된 표준 편차 : ", new_B)
print()
print()
high_count = 0
high_score = 0
first_score = 0
high_score_list = []
high_final_list = []
count = 2
while count < 12:
    final_list = []
    for i in range(count):

        final_list.append(std_list.index(new_B[i]))

    new_train_col = train_col.iloc[:, final_list]

    #print(new_train_col)

    scaler = MinMaxScaler()

    scaler.fit_transform(new_train_col)

    A = new_train_col.to_numpy()

    #print(A)

    km = KMeans(n_clusters=3)
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
    if count == 11:
        first_score = score
    count += 1
print("초기 데이터셋 점수 :", first_score) # 초기 데이터셋 점수
print("가장 높은 점수 :", high_score) # 가장 높은 점수
print("선택된 피처 인덱스 :", high_final_list) # 선택된 피처 인덱스
print(high_score_list) # 가장 높은 점수의 데이터셋
print("점수 증가율 :", (high_score - first_score) / first_score * 100)
km = KMeans(n_clusters=3)
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

sns.scatterplot(x=X[:,0], y=X[:, 1], c=km.labels_)

#sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_, data=pca_tra)
#sns.pairplot(new_X)
plt.show()
