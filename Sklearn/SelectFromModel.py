from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import scipy.io
import pandas as pd

mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
new_mat_file_value = pd.DataFrame(mat_file_value)
print(mat_file_label)
mat_file_label = pd.DataFrame(mat_file_label)

selector = SelectFromModel(estimator=LogisticRegression()).fit(mat_file_value, mat_file_label)

print(selector.estimator_.coef)
print(selector.threshold_)
print(selector.get_support())
X = selector.transform(mat_file_value)

print(X.shape)