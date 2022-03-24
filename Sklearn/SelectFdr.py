from sklearn.feature_selection import SelectFdr, chi2
import scipy.io
import pandas as pd

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
new_mat_file_value = pd.DataFrame(mat_file_value)

X = SelectFdr(chi2, alpha=0.01).fit_transform(mat_file_value, mat_file_label)

print(X.shape)
# 수정 필요
