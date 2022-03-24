import copy
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
import random
import scipy.io

start = timer()
mat_file_name = "../DataSet/YaleB_32x32.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]
new_mat_file_value = pd.DataFrame(mat_file_value)

km = KMeans(n_clusters=3)
km.fit(new_mat_file_value)
centers = km.cluster_centers_
clusters = km.predict(new_mat_file_value)
first_score = silhouette_score(new_mat_file_value, clusters)

Best_DataSet = []  # feature 데이터셋
Best_Score = 0  # Silhouette score 생성
Best_Feature_Index = []  # feature index 리스트
Population = 50
mutation = 0.02


def Best10List(List): # 문제 수정

    countList = [0 for __ in range(Population)]
    countListIndex = []
    AnotherList = []
    for i in range(Population):
        zerocount = 0
        for j in range(1024):

            if List[i][j] == 1:
                zerocount += 1
        countList[i] = zerocount
    #print("Best 10 List 실험", countList)
    print("평균값 : ", int(sum(countList)/Population))

    for i in range(Population):
        if countList[i] <= int(sum(countList)/Population):
            countListIndex.append(i)
    print(countListIndex)
    print("countListIndex 길이 : ", len(countListIndex))
    for i in range(Population):
        if i in countListIndex:
            AnotherList.append(List[i])
    #print("AnotherList : ", AnotherList)
    print("AnotherList 길이: ", len(AnotherList))
    return AnotherList


def randomList(list):
    count = 60
    if len(list) < count:
        while True:
            A = random.randint(0, 1023)
            if A not in list:
                list.append(A)
            if len(list) == count:
                break
    elif len(list) > count:
        while True:
            list.remove(random.choice(list))
            if len(list) == count:
                break
    else:
        pass
    return sorted(list)


def evaluation(List):
    global Best_Score, Best_DataSet, Best_Feature_Index
    start1 = timer()
    score_list = []
    AnotherList = Best10List(List)
    Feature_index = [[] for __ in range(len(AnotherList))]
    for i in range(len(AnotherList)):
        for j in range(1024):
            if AnotherList[i][j] == 1:
                Feature_index[i].append(j)
    #print("의심 구간1 : ", Feature_index)
    #print(len(Feature_index))
    for i in range(len(Feature_index)):
        Feature_index[i] = randomList(Feature_index[i])

    #print("의심 구간2 : ", Feature_index)
    #print(len(Feature_index))
    for i in Feature_index:
        #print("피처수 확인 : ", len(i))
        if len(i) == 0:
            print("1 없음")
            #print(i)
            continue
        else:
            new_train_col = new_mat_file_value.iloc[:, i]
            km = KMeans(n_clusters=3)
            km.fit(new_train_col)
            clusters = km.predict(new_train_col)
            score = silhouette_score(new_train_col, clusters)
            score_list.append(score)
            np_new_train_col = new_train_col.to_numpy()

            if score > Best_Score:
                print(len(np_new_train_col[0]))  # 피처의 수 확인
                Best_Score = score
                print(Best_Score)
                Best_DataSet = new_train_col
                Best_Feature_Index = i
    end1 = timer()
    print("수정한 평가 메소드 소요시간 : ", timedelta(seconds=end1 - start1))

    return score_list


def MutationFunc(List):
    for i in range(1024):
        if List[i] == 1 and random.random() < 0.08:
            List[i] = 0
            #print("확률!!")

    return List


List = [[0 for _ in range(1024)] for _ in range(Population)]

new_List = copy.deepcopy(List)

good_parent_index = [0, 0]

count = 0

for i in range(Population):
    for j in range(len(List[0])):
        List[i][j] = random.randint(0, 1)
count += 1
end = timer()
# print("첫 부분 소요 시간 : ", timedelta(seconds=end-start))

while True:

    Score_list = evaluation(List)
    print("Score_list : ", Score_list)
    start = timer()
    sort_Score_list = np.sort(Score_list)[::-1]
    set_sort_Score_list = sorted(set(sort_Score_list))
    # sort_Score_list = sorted(list(set_sort_Score_list))
    sort_Score_list = list(reversed(set_sort_Score_list))
    print("스코어 리스트 : ", Score_list)
    print("정렬된 스코어 리스트 : ", sort_Score_list)

    if len(sort_Score_list) <= 1:
        mutation = 0.1
        for i in range(2):
            good_parent_index[i] = Score_list.index(sort_Score_list[0])

    else:
        for i in range(2):  # 최상위 부모 인덱스 리스트에서 2개 랜덤 추출
            good_parent_index[i] = Score_list.index(sort_Score_list[i])
        print(good_parent_index)

    for i in range(Population):  # CrossOver
        Division_Point = random.randint(1, 1021)
        Parent_1 = List[good_parent_index[0]][:Division_Point]
        Parent_2 = List[good_parent_index[1]][Division_Point:1024]
        new_List[i] = Parent_1 + Parent_2

    for i in range(Population):  # Mutation
        if random.random() < mutation:
            print("돌연변이 발생", mutation)
            # print(new_List[i])
            new_List[i] = MutationFunc(new_List[i])
            # print(new_List[i])
    if mutation == 0.1:
        mutation = 0.02
    List = copy.deepcopy(new_List)
    count += 1
    print("count : ", count)
    end = timer()
    print("현재 베스트 피처 수 : ", len(Best_Feature_Index))
    print("현재 최고 점수 : ", Best_Score)
    # print("while문 소요시간 : ", timedelta(seconds=end - start))

    if count > 100 or len(Best_Feature_Index) == 1:
        break


# print(Score_list)
print(first_score)
print("최고 점수", Best_Score)
print(Best_Feature_Index)
print((Best_Score - first_score) / first_score * 100)
print("")
print(len(Best_Feature_Index))

if len(Best_Feature_Index) == 1:
    new_train_col = new_mat_file_value.iloc[:, Best_Feature_Index]
    km = KMeans(n_clusters=3)
    km.fit(new_train_col)
    sns.scatterplot(x=new_mat_file_value.iloc[:, 0], y=new_train_col.iloc[:, 0], c=km.labels_)
    plt.show()

else:
    Best_DataSet = Best_DataSet.to_numpy()
    km = KMeans(n_clusters=3)
    km.fit(Best_DataSet)
    pca = PCA(n_components=2)
    pca_tra = pca.fit_transform(Best_DataSet)  # pca 적용할 때
    # sns.scatterplot(x=Best_DataSet[:, 0], y=Best_DataSet[:, 1], c=km.labels_)
    sns.scatterplot(x=pca_tra[:, 0], y=pca_tra[:, 1], c=km.labels_)  # pca 적용할 때
    plt.show()
