import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from pandas.plotting import scatter_matrix
import numpy as np
import scipy
import sklearn as sl
from scipy.stats import describe

data = pd.read_csv('cleaned_cleveland.csv',index_col=[0])


df = pd.DataFrame(data)

meanList = df.mean()

df_0 = pd.DataFrame(data)
print(df_0)

for i in df.columns:
   df_0[i] = df[i] - meanList[i]

print(df_0)

df_normalized = pd.DataFrame(df_0)

for i in df_normalized.columns:
    df_normalized[i] = df_normalized[i]/df_normalized[i].std(axis=0)

print(df_normalized.describe())


plt.matshow(df_normalized.corr(), cmap='seismic', vmin = -1, vmax = 1)
plt.yticks(range(df_normalized.select_dtypes(['number']).shape[1]), df_normalized.select_dtypes(['number']).columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=12)


scatter_matrix(df_normalized,alpha=0.2, figsize=(15,15), diagonal='kde')

plt.show()
