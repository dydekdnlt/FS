from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
import scipy.io
import pandas as pd
from sklearn.svm import SVR

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
new_mat_file_value = pd.DataFrame(mat_file_value)
new_mat_file_label = pd.DataFrame(mat_file_label)

np_new_mat_file_label = new_mat_file_label.values
np_new_mat_file_label = np_new_mat_file_label.ravel()
print(np_new_mat_file_label)
print("")


estimator = KNeighborsClassifier(n_neighbors=5)
#estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1000, step=1)
selector = selector.fit(new_mat_file_value, np_new_mat_file_label)
print(selector)

# 레이블 차원 줄이기ravel() 사용