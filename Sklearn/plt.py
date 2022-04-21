import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# YaleB, ORL 분류오류율, 표준편차 수정
a = [29.54, 27.94, 12.04, 20.18, 38.22, 16.39, 34.08, 38.98, 37.37] # SKB
b = [31.77, 25.38, 14.53, 20.77, 43.35, 19.36, 72.49, 41.99, 52.61] # VT(std)
c = [25.55, 24.86, 11.58, 15.8, 45.57, 45.35, 22.18, 29.49, 100] # MRMR
Dataset = ['Wine', 'Hepatitis', 'Ionosphere', 'Spambase', 'Arrhythmia', 'Madelon', 'YaleB_32x32', 'ORL_32x32', 'Yale_64x64']

df = pd.DataFrame({'SKB' : a, 'VT(std)' : b, 'MRMR' : c}, index=Dataset)
print(df)

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25

index = np.arange(9)

b1 = plt.bar(index, df['SKB'], bar_width, alpha=0.4, color='red', label='SKB')
b2 = plt.bar(index + bar_width, df['VT(std)'], bar_width, alpha=0.4, color='blue', label='VT(std)')
b3 = plt.bar(index + 2 * bar_width, df['MRMR'], bar_width, alpha=0.4, color='green', label='MRMR')

plt.xticks(np.arange(bar_width, 9 + bar_width, 1), Dataset)

plt.xlabel('Dataset', size = 13)
plt.ylabel('CER(%)', size = 13)
plt.legend()
plt.show()