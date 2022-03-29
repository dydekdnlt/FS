import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR

train = pd.read_csv("../../DataSet/spambase.csv", header=None)
label = pd.DataFrame(train[57])
label = label.values
label = label.ravel()
value = np.delete(np.array(train), 57, axis=1)
value = pd.DataFrame(value)
print(label.shape)
print(value.shape)

estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=24, step=1)
X = selector.fit_transform(value, label)
print(X)
print(X.shape)

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

