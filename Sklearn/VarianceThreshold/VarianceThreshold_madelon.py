import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import time

start = time.time()

train = pd.read_csv("../../DataSet/madelon.csv", header=0, index_col=0)
label = np.array(train['500'])
value = np.delete(np.array(train), label, axis=1)
value = pd.DataFrame(value)
print(label.shape)
print(value.shape)
print("레이블 : ", label)
print("패턴 : ", value)

std_list = []
sort_std_list = []
for i in value:
    std_list.append(np.std(value[i]))

# print("표준편차 : ", std_list)

sort_std_list.append(sorted(std_list))
reverse_sort_std_list = sort_std_list[0]

reverse_sort_std_list.sort(reverse=True)
print("정렬된 표준 편차 : ", reverse_sort_std_list)

index_list = []

for i in range(40):
    index_list.append(std_list.index(reverse_sort_std_list[i]))
    print(std_list.index(reverse_sort_std_list[i]))
X = value.iloc[:, index_list]

print(X.shape)

new_clf = KNeighborsClassifier(n_neighbors=3)
new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(X, label, test_size=0.25)
new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=2)
# print("new score : ", new_scores)
# print("new mean accuracy of validation : ", np.mean(new_scores))
new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
new_Y_pred = new_clf.predict(new_X_test)
print(new_Y_pred)
print(new_Y_test.ravel())
minScore = 1 - accuracy_score(new_Y_test.ravel(), new_Y_pred)
print(minScore)

end = time.time()

print(f"{end - start: .5f} sec")
