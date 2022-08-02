import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)

FS = ['SKB', 'VT(std)', 'MRMR', 'MCFS', 'LS', 'NDFS']
value = [16.39, 19.36, 45.35, 20.28, 100, 45.21]

plt.bar(x, value)
plt.xticks(x, FS)
plt.ylim([0, 100])
plt.title('Madelon')

plt.show()