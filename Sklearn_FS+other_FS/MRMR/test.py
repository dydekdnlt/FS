from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bins):
    c = np.histogram2d(x, y, bins)
    print("c : ", c)
    c_xy = np.histogram2d(x, y, bins)[0]
    print("c_xy : ", c_xy)
    mi = mutual_info_score(None, None, contingency=c_xy)
    print(mi)
    return c_xy


def calc_MI_2(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    print(mi)
    return mi



train = pd.read_csv("../../DataSet/wine.csv", header=None)
label = np.array(train[0])
value = np.delete(np.array(train), 0, axis=1)


y = label.ravel()
print(y)
x = pd.DataFrame(value).iloc[:, 0]
print(x)
x_2 = pd.DataFrame(value).iloc[:, 1]
x_3 = pd.DataFrame(value).iloc[:, 2]
A = calc_MI(x, y, 3)
#B = calc_MI(x_2, y, 10)
#C = calc_MI(x_3, y, 10)
# plt.hist(x, bins=10)
plt.hist(A, bins=10)
#plt.hist(B, bins=10)
#plt.hist(C, bins=10)
plt.show()