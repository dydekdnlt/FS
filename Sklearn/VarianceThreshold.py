from sklearn.feature_selection import VarianceThreshold
import scipy.io
import pandas as pd


mat_file_name = "../DataSet/Yale_64x64.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]
new_mat_file_value = pd.DataFrame(mat_file_value)

print(mat_file_value)
print(len(mat_file_value))
print("")
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
update_sel = sel.fit_transform(mat_file_value)

print(len(update_sel))
