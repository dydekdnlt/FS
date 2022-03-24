from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import scipy.io
import pandas as pd

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
new_mat_file_value = pd.DataFrame(mat_file_value)
new_mat_file_label = pd.DataFrame(mat_file_label[0])


knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=500)
sfs.fit(mat_file_value, new_mat_file_label)
print(sfs.transform(mat_file_value).shape)

# 수정 필요