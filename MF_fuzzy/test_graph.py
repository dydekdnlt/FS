import numpy as np
import matplotlib.pyplot as plt
t = np.arange(4, 14, 1)
A = [0.7402777777777779, 0.8444444444444443, 0.8597222222222222, 0.8444444444444444, 0.9166666666666664, 0.9430555555555555, 0.8944444444444443, 0.9319444444444442, 0.8527777777777781, 0.7222222222222221]
B = [0.7083333333333333, 0.6833333333333333, 0.7194444444444444, 0.673611111111111, 0.7180555555555557, 0.6888888888888888, 0.6861111111111111, 0.698611111111111, 0.7000000000000001, 0.7097222222222224]
b = np.mean(B)
C = [b for i in range(10)]
print(len(A), len(B))
print(b)

plt.xlim(3, 14)
plt.ylim(0.6, 1.0)
plt.xlabel("Number of selected features")
plt.ylabel("Accuracy Score")
plt.plot(t, A, label='LS+SVD')
plt.plot(t, C, label='SVD')
plt.title('SVD_wine')
plt.legend(loc='upper right')
plt.show()
