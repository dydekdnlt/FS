import copy

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

train = pd.read_csv("C:\\Users\\용다윗\\Desktop\\perfume_data.csv")
# print(train)

train_row = train[['name']]
col_name = ['1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28']
train_col = train[col_name]

new_col_name = col_name
A = train_col.to_numpy()
km = KMeans(n_clusters=3)
km.fit(A)
centers = km.cluster_centers_
clusters = km.predict(A)
first_score = silhouette_score(A, clusters)
Best_Col = []
Best_Score = 0
Best_index = []
Best_index_feature = []
size = 10
Tabu_Queue = [] * size

First_List = [0 for _ in range(len(col_name))]
for i in range(len(First_List)):
    First_List[i] = random.randint(0, 1)
new_List = copy.deepcopy(First_List)

k = 0


def ListToFeature(List):  # 0, 1 리스트 -> feature index 리스트 메소드 완료
    FeatureIndex = []
    for i in range(len(List)):
        if List[i] == 1:
            FeatureIndex.append(i)
    return FeatureIndex


def ObjectiveFunction(List):  # 점수내는 메소드 완료

    A = ListToFeature(List)
    if len(A) == 0:
        score = 0
    else:
        new_train_col = train_col.iloc[:, A]
        km = KMeans(n_clusters=3)
        km.fit(new_train_col)
        clusters = km.predict(new_train_col)
        score = silhouette_score(new_train_col, clusters)

    return score


def NeighborSearch(List):  # 주변 피처 리스트 찾기 완료
    Trial_solution = []

    for z in range(len(List)):
        copy_List = List.copy()
        if copy_List[z] == 1:
            copy_List[z] = 0
            Trial_solution.append(copy_List)
        else:
            copy_List[z] = 1
            Trial_solution.append(copy_List)

    return Trial_solution


def RandomChoice(List):
    if int(len(col_name)) > 2:
        RandomChoiceList = random.sample(List, int(len(col_name) / 2))
    else:
        RandomChoiceList = List
    return RandomChoiceList


def BestValue(List):
    max_ob = []
    for i in List:
        max_ob.append(ObjectiveFunction(i))
    max_ob_value = max(max_ob)
    max_ob_index = max_ob.index(max_ob_value)
    return max_ob_index, max_ob_value


result = NeighborSearch(First_List)
Random_result = RandomChoice(result)

new_Random_result = copy.deepcopy(Random_result)
#print("최고인덱스, 값", BestValue(new_Random_result))
while k < 200:
    if len(Random_result) == 0:
        result = NeighborSearch(Best_index)
        Random_result = RandomChoice(result)
        print(Random_result)
        print("재부팅")

    else:
        print(len(Random_result))
        max_ob_index, max_ob_value = BestValue(Random_result)

        print("랜덤리스트 최고값 :", max_ob_value)
        if max_ob_value > Best_Score and Random_result[max_ob_index] not in Tabu_Queue:
            Best_Score = max_ob_value
            Best_index = Random_result[max_ob_index]
            if len(Tabu_Queue) >= size:
                Tabu_Queue.pop(0)
            Tabu_Queue.append(Random_result[max_ob_index])
            print("타부 크기", len(Tabu_Queue))
            Random_result.remove(Random_result[max_ob_index])
            print("삭제1 랜덤리스트", Random_result)
            print("삭제1")

        elif max_ob_value > Best_Score > Best_Score and Random_result[max_ob_index] in Tabu_Queue:
            Best_Score = max_ob_value
            Best_index = Random_result[max_ob_index]

            if len(Tabu_Queue) >= size:
                Tabu_Queue.pop(0)
            Tabu_Queue.append(Random_result[max_ob_index])
            print("타부 크기", len(Tabu_Queue))
            Random_result.remove(Random_result[max_ob_index])
            print("삭제2 랜덤리스트", Random_result)
            print("삭제 2")
        elif Random_result[max_ob_index] in Tabu_Queue and ObjectiveFunction(Random_result[max_ob_index]) <= Best_Score:

            Random_result.remove(Random_result[max_ob_index])

            print("삭제3 랜덤리스트", Random_result)
            print("삭제 3")
        else:
            Random_result.remove(Random_result[max_ob_index])
            print("삭제4 랜덤리스트", Random_result)
            print("삭제 4")

    k += 1

print("")
print(first_score)
print("최고 점수 :", Best_Score)
print(Best_index)

# print(Best_index_feature)
'''
if len(tabu_list) >= size:
            tabu_list.pop(0)
'''
A = ListToFeature(Best_index)
print(len(A))
if len(A) == 1:

    new_train_col = train_col.iloc[:, A]
    print("train row: ", train_row)
    print("new train col :", new_train_col)
    km = KMeans(n_clusters=3)
    km.fit(new_train_col)
    sns.scatterplot(x=train_row.iloc[:, 0], y=new_train_col.iloc[:, 0], c=km.labels_)
    plt.show()
else:
    new_train_col = train_col.iloc[:, A]
    km = KMeans(n_clusters=3)
    km.fit(new_train_col)
    centers = km.cluster_centers_
    Cluster = new_train_col.copy()
    X = Cluster
    pca = PCA(n_components=2)
    pca_tra = pca.fit_transform(X)
    sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_)
    plt.show()