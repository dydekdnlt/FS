from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
import scipy.io
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블

# new_mat_file_value = pd.DataFrame(mat_file_value)
# print(mat_file_label)
# mat_file_label = pd.DataFrame(mat_file_label)


clf = KNeighborsClassifier(n_neighbors=3)

selector = SelectFromModel(estimator=RandomForestRegressor(n_estimators=10), threshold="median")

X_select = selector.fit_transform(mat_file_value, mat_file_label.ravel())
print(X_select)
print(len(X_select[0]))
print("")
print(mat_file_value)
print(len(mat_file_value[0]))

X_train, X_test, Y_train, Y_test = train_test_split(mat_file_value, mat_file_label, test_size=0.25)
scores = cross_val_score(clf, X_train, Y_train.ravel(), cv=5)
print(scores)
print("mean accuracy of validation: ", np.mean(scores))
clf = clf.fit(X_train, Y_train.ravel())
Y_pred = clf.predict(X_test)
print(Y_pred)
print(Y_test.ravel())
print(accuracy_score(Y_test.ravel(), Y_pred))

'''
selector = SelectFromModel(estimator=LogisticRegression()).fit(mat_file_value, mat_file_label)

print(selector.estimator_.coef)
print(selector.threshold_)
print(selector.get_support())
X = selector.transform(mat_file_value)

print(X.shape)
'''

