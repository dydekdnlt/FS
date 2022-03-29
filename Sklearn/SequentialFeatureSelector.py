from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import scipy.io
import pandas as pd

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
new_mat_file_value = pd.DataFrame(mat_file_value)
new_mat_file_label = pd.DataFrame(mat_file_label)

np_new_mat_file_label = new_mat_file_label.values
np_new_mat_file_label = np_new_mat_file_label.ravel()

knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=500)
X = sfs.fit_transform(mat_file_value, np_new_mat_file_label)
print(X.shape)

# 수정 필요