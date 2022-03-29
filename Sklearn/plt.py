import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


a = [29.54, 20.18, 34.08, 38.98, 37.37]
b = [7.06, 0, 0, 0, 40.94]
c = [31.77, 20.77, 72.49, 41.99, 52.61]
Dataset = ['Wine', 'Spambase', 'YaleB_32x32', 'ORL_32x32', 'Yale_64x64']

df = pd.DataFrame({'SKB' : a, 'RFE' : b, 'VT(std)' : c}, index = Dataset)
print(df)

fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25

index = np.arange(5)

b1 = plt.bar(index, df['SKB'], bar_width, alpha=0.4, color='red', label='SKB')
b2 = plt.bar(index + bar_width, df['RFE'], bar_width, alpha=0.4, color='blue', label='RFE')
b3 = plt.bar(index + 2 * bar_width, df['VT(std)'], bar_width, alpha=0.4, color='green', label='VT(std)')

plt.xticks(np.arange(bar_width, 5 + bar_width, 1), Dataset)

plt.xlabel('Dataset', size = 13)
plt.ylabel('CER(%)', size = 13)
plt.legend()
plt.show()