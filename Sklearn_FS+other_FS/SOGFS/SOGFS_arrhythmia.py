import numpy as np
import pandas as pd
import scipy.io
from sklearn.impute import SimpleImputer

train = pd.read_csv("../../DataSet/arrhythmia.csv", header=None)
imputer = SimpleImputer(strategy="mean")
train = pd.DataFrame(imputer.fit_transform(train))
label = np.array(train[279])
value = np.delete(np.array(train), 279, axis=1)
print(value)


def get_Laplacian(M):
    S = M.sum(1)
    i_nz = np.where(S > 0)[0]
    S = S[i_nz]
    M = (M[i_nz].T)[i_nz].T
    S = 1 / np.sqrt(S)
    M = S * M
    M = (S * M.T).T
    n = np.size(S)
    M = np.identity(n) - M
    M = (M + M.T) / 2
    return M


LS = get_Laplacian(value)
print(LS.shape)