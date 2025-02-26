import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

def normalize_data(df):
    # We extract a list of mean values for each column
    meanList = df.mean()
    # We create an empty dataframe
    df_0 = pd.DataFrame()

    #For each column, we subtract the corresponding mean
    for i in df.columns:
        df_0[i] = df[i] - meanList[i]

    #We create a copy of the meanLess dataframe
    df_normalized = pd.DataFrame(df_0)

    #We divide each column with the corresponding STD.
    for i in df_normalized.columns:
        df_normalized[i] = df_normalized[i]/df_normalized[i].std(axis=0)
    return df_normalized


def plot_heatmap(df):
    plt.figure(figsize=(10,10),dpi=300)
    sns.heatmap(df.corr(), cmap = 'seismic', vmax = 1, vmin = -1,
            square = True, linewidths=1, annot=True, annot_kws={"size": 10})
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')


def plot_scatter_matrix(df):
    plt.figure(figsize = (20,20), dpi = 300)
    scatter_matrix(df,alpha=0.2, figsize=(15,15), diagonal='kde')
    plt.tight_layout()
    plt.savefig('scatter_matrix.png')

data = pd.read_csv('cleaned_cleveland.csv',index_col=[0])
# Firstly, we load in the data into a dataframe.
df_normal = normalize_data(data)

plot_heatmap(df_normal)