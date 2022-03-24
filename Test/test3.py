import copy
from timeit import default_timer as timer
from datetime import timedelta
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

mat_file_name = "../DataSet/YaleB_32x32.mat"
mat_file = scipy.io.loadmat(mat_file_name)

print(type(mat_file))

for i in mat_file:
    print(i)

mat_file_value = mat_file["fea"]
print(len(mat_file_value), len(mat_file_value[0]))
print(mat_file_value)
new_mat_file_value = pd.DataFrame(mat_file_value)
# A = mat_file_value.to_numpy()
km = KMeans(n_clusters=3)
km.fit(new_mat_file_value)
centers = km.cluster_centers_
clusters = km.predict(new_mat_file_value)
first_score = silhouette_score(new_mat_file_value, clusters)
Best_Col = []
Best_Score = 0
Best_name = []


def evalution(List):
    global Best_Score, Best_Col, Best_name
    start = timer()
    Score_list = []
    Feature_index = [[] for __ in range(50)]
    for i in range(50):
        for j in range(1024):
            if List[i][j] == 1:
                Feature_index[i].append(j)
    # print(Feature_index)
    for i in Feature_index:
        new_train_col = new_mat_file_value.iloc[:, i]
        km = KMeans(n_clusters=3)
        km.fit(new_train_col)
        clusters = km.predict(new_train_col)
        score = silhouette_score(new_train_col, clusters)
        Score_list.append(score)
        np_new_train_col = new_train_col.to_numpy()

        if score > Best_Score and len(np_new_train_col[0]) >= 1:
            print(len(np_new_train_col[0]))
            Best_Score = score
            Best_Col = new_train_col
            Best_name = i
    end = timer()
    # print(Best_Col)
    # print("")
    # print(Best_Score)
    print("평가 메소드 수행 시간 : ", timedelta(seconds=end - start))
    return Score_list


List = [[0 for _ in range(1024)] for _ in range(50)]

new_List = copy.deepcopy(List)
mutation = 0.01

good_parent_index = [0, 0]

count = 0

for i in range(50):
    for j in range(len(List[0])):
        List[i][j] = random.randint(0, 1)
count += 1

while True:

    Score_list = evalution(List)
    sort_Score_list = np.sort(Score_list)[::-1]

    for i in range(2):
        good_parent_index[i] = Score_list.index(sort_Score_list[i])
    for i in range(50):
        for j in range(len(List[0])):
            if random.random() < mutation:

                if new_List[i][j] == 1 and new_List[i].count(1) >= 2 and List[i].count(1) >= 2:
                    new_List[i][j] = 0
                elif new_List[i][j] == 0:
                    new_List[i][j] = 1
                else:
                    pass

            else:
                new_List[i][j] = List[good_parent_index[random.randint(0, 1)]][j]

    List = copy.deepcopy(new_List)
    count += 1
    print("count : ", count)
    if count > 1000:
        break

# print(Score_list)
print(first_score)
print("최고 점수", Best_Score)
print(Best_name)
print((Best_Score - first_score) / first_score * 100)
print("")
print(len(Best_name))

Best_Col = Best_Col.to_numpy()

km = KMeans(n_clusters=3)
km.fit(Best_Col)
pca = PCA(n_components=2)
pca_tra = pca.fit_transform(Best_Col)  # pca 적용할 때
# sns.scatterplot(x=Best_Col[:, 0], y=Best_Col[:, 1], c=km.labels_)
sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_)  # pca 적용할 때
plt.show()
