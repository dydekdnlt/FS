from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
import scipy.io
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import plotly.express as px

mat_file_name = "../../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"] # 패턴
mat_file_label = mat_file["gnd"] # 레이블

# YaleB 데이터셋 : 470 클래스 수 : 38
# ORL 데이터셋 : 678 클래스 수 : 40
# Yale_64x64 데이터셋 : 388 , 클래스 수 : 15

X = SelectKBest(chi2, k=388).fit_transform(mat_file_value, mat_file_label)
# print(mat_file_value.shape)
# print(X)
# print(X.shape)

clf = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, Y_train, Y_test = train_test_split(mat_file_value, mat_file_label, test_size=0.25)
scores = cross_val_score(clf, X_train, Y_train.ravel(), cv=2)
# print(scores)
# print("mean accuracy of validation: ", np.mean(scores))
clf = clf.fit(X_train, Y_train.ravel())
Y_pred = clf.predict(X_test)
# print(Y_pred)
# print(len(Y_pred))
# print(Y_test.ravel())
# print(len(Y_test.ravel()))
# print(1 - accuracy_score(Y_test.ravel(), Y_pred))

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(X, mat_file_label, test_size=0.25)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
# print("new score : ", new_scores)
# print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
# print(new_Y_pred)
# print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print(minScore)


