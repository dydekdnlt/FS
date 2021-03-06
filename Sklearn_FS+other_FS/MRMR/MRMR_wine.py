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

warnings.filterwarnings(action='ignore')

start = time.time()
# 완료
train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = np.array(train[0])
value = np.delete(np.array(train), 0, axis=1)
print(value.shape)
Score_List = []
lr = label.ravel()


def calc_MI(x, y):
    c_xy = np.histogram2d(x, y, 2)[0]

    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def Mutual_Info_Score(x):
    A = mutual_info_score(pd.DataFrame(value).iloc[:, x], lr)
    return A


def Argmax_List(x):
    A = max(x)
    A_index = x.index(A)
    return A_index


for i in range(13):
    # print("Mutual Information", i, Mutual_Info_Score(i))
    print("Mutual Information", i, calc_MI(pd.DataFrame(value).iloc[:, i], lr))
    # Score_List.append(Mutual_Info_Score(i))
    Score_List.append(calc_MI(pd.DataFrame(value).iloc[:, i], lr))

print(Score_List)
print(Argmax_List(Score_List))

subFeature_index = [Argmax_List(Score_List)]
subFeature = pd.DataFrame(value).iloc[:, Argmax_List(Score_List)]
new_subFeature = copy.copy(subFeature)
value = np.delete(value, Argmax_List(Score_List), axis=1)
new_value = copy.copy(value)
print(subFeature.shape)
K = 1
while K < 6:
    next_score_list = []
    new_subFeature = np.array(new_subFeature)
    new_subFeature = pd.DataFrame(new_subFeature)
    for i in range(len(pd.DataFrame(new_value[0]))):
        another_score_list = []
        for j in range(K):
            another_score_list.append(
                calc_MI(pd.DataFrame(new_value).iloc[:, i], pd.DataFrame(new_subFeature).iloc[:, j]))
                # mutual_info_score(pd.DataFrame(new_value).iloc[:, i], pd.DataFrame(new_subFeature).iloc[:, j]))

        next_score_list.append(calc_MI(pd.DataFrame(value).iloc[:, i], lr) - ((1 / K) * sum(another_score_list)))
    print(i, next_score_list)
    print(next_score_list.index(max(next_score_list)))
    new_subFeature = pd.concat([new_subFeature, pd.DataFrame(new_value).iloc[:, next_score_list.index(max(next_score_list))]], axis=1)
    # print(max(next_score_list))
    # subFeature = np.concatenate((subFeature, pd.DataFrame(value).iloc[:, next_score_list.index(max(next_score_list))]), axis=1)
    # subFeature = subFeature.assign(pd.DataFrame(value).iloc[:, next_score_list.index(max(next_score_list))])
    # print(subFeature)
    new_value = np.delete(new_value, next_score_list.index(max(next_score_list)), axis=1)
    K += 1
    print("new_subFeature shape : ", new_subFeature.shape)
    print("new_value shape : ", new_value.shape)

print(new_subFeature)
new_subFeature = np.array(new_subFeature)
new_subFeature = pd.DataFrame(new_subFeature)

print(new_subFeature)


new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_subFeature, label, test_size=0.50)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print("mrmr 스코어 : ", minScore)
end = time.time()
print(f"{end - start: .5f} sec")
