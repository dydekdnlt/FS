import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
import time

start = time.time()

train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = pd.DataFrame(train[0])
label = label.values
label = label.ravel()
value = np.delete(np.array(train), 0, axis=1)
value = pd.DataFrame(value)
print(label.shape)
print(value.shape)
print("레이블 : ", label)
print("패턴 : ", value)

'''
label = np.array(train[[0]])
value = np.delete(np.array(train), 0, axis=1)
print(label)
print(value)
print(len(label))
print(len(value))
# np.reshape(value, (-1, 1))
'''
estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=6, step=1)
X = selector.fit_transform(value, label)
print(X)
print(X.shape)
'''
clf = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, Y_train, Y_test = train_test_split(value, label, test_size=0.25)
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
'''
new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(X, label, test_size=0.25)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
print("new score : ", new_scores)
print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print(minScore)
end = time.time()
print(f"{end - start: .5f} sec")
