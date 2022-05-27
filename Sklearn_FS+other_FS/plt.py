import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 그래프 위치 수정

a = [29.54, 27.94, 12.04, 20.18, 38.22, 16.39, 34.08, 38.98, 37.37]  # SKB
b = [31.77, 25.38, 14.53, 20.77, 43.35, 19.36, 72.49, 41.99, 52.61]  # VT(std)
c = [31.13, 26.17, 12.38, 11.6, 42.64, 45.16, 26.32, 31.49, 100]  # MRMR
d = [6.44, 21.53, 14.31, 13.3, 43.79, 20.28, 42.49, 18.69, 42.13]  # MCFS
Dataset = ['Wine', 'Hepatitis', 'Ionosphere', 'Spambase', 'Arrhythmia', 'Madelon', 'YaleB_32x32', 'ORL_32x32', 'Yale_64x64']

df = pd.DataFrame({'SKB' : a, 'VT(std)' : b, 'MRMR' : c, 'MCFS' : d}, index=Dataset)
print(df)

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.20

index = np.arange(9)

b1 = plt.bar(index, df['SKB'], bar_width, alpha=0.4, color='red', label='SKB')
b2 = plt.bar(index + bar_width, df['VT(std)'], bar_width, alpha=0.4, color='blue', label='VT(std)')
b3 = plt.bar(index + 2 * bar_width, df['MRMR'], bar_width, alpha=0.4, color='green', label='MRMR')
b4 = plt.bar(index + 3 * bar_width, df['MCFS'], bar_width, alpha=0.4, color='black', label='MCFS')


plt.xticks(np.arange(bar_width, 9 + bar_width, 1), Dataset)

plt.xlabel('Dataset', size = 13)
plt.ylabel('CER(%)', size = 13)
plt.legend()
plt.show()
'''

a = [31, 29.09, 20.96, 20.6, 43.09, 100, 61.28, 42.23, 53.61]  # LS
b = [31.77, 25.38, 14.53, 20.77, 43.35, 19.36, 72.49, 41.99, 52.61]  # VT(std)

Dataset = ['Wine', 'Hepatitis', 'Ionosphere', 'Spambase', 'Arrhythmia', 'Madelon', 'YaleB_32x32', 'ORL_32x32', 'Yale_64x64']

df = pd.DataFrame({'LS' : a, 'VT(std)' : b}, index=Dataset)
print(df)

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.20

index = np.arange(9)

b1 = plt.bar(index, df['LS'], bar_width, alpha=0.4, color='red', label='LS')
b2 = plt.bar(index + bar_width, df['VT(std)'], bar_width, alpha=0.4, color='blue', label='VT(std)')
#b3 = plt.bar(index + 2 * bar_width, df['MRMR'], bar_width, alpha=0.4, color='green', label='MRMR')
#b4 = plt.bar(index + 3 * bar_width, df['MCFS'], bar_width, alpha=0.4, color='black', label='MCFS')


plt.xticks(np.arange(bar_width, 9 + bar_width, 1), Dataset)

plt.xlabel('Dataset', size = 13)
plt.ylabel('CER(%)', size = 13)
plt.legend()
plt.show()
'''