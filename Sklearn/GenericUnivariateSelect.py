from sklearn.feature_selection import GenericUnivariateSelect, chi2
import scipy.io
import pandas as pd

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]
mat_file_label = mat_file["gnd"]
new_mat_file_value = pd.DataFrame(mat_file_value)

print(mat_file_value)
print(len(mat_file_value))
print("")

transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)

X = transformer.fit_transform(mat_file_value, mat_file_label)
print(X)
print(len(X))

# 수정 필요
