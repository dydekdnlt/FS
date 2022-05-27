import numpy as np
import pandas as pd
import scipy.io

mat_file_name = "../../DataSet/ORL_32x32.mat"
mat_file = scipy.io.loadmat(mat_file_name)
mat_file_value = mat_file["fea"]  # 패턴
mat_file_label = mat_file["gnd"]  # 레이블
mat_file_value = mat_file_value.astype(float)
value = np.array(mat_file_value)
label = np.array(mat_file_label)

print(value)
def get_Laplacian(M):
    S = M.sum(1)
    i_nz = np.where(S > 0)[0]
    S = S[i_nz]
    M = M[i_nz].T[i_nz].T
    S = 1 / np.sqrt(S)
    M = S * M
    M = (S * M.T).T
    n = np.size(S)
    M = np.identity(n) - M
    M = (M + M.T) / 2
    return M


LS = get_Laplacian(value)
print(LS.shape)