import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def normalize_data(df):
    # We extract a list of mean values for each column
    mean_list = df.mean()
    # We create an empty dataframe
    df_0 = pd.DataFrame()

    #For each column, we subtract the corresponding mean
    for i in df.columns:
        df_0[i] = df[i] - mean_list[i]

    #We create a copy of the meanLess dataframe
    df_normalized = pd.DataFrame(df_0)

    #We divide each column with the corresponding STD.
    for i in df_normalized.columns:
        df_normalized[i] = df_normalized[i]/df_normalized[i].std(axis=0)


    print(df_normalized.describe())
    return df_normalized


def extract_relevant_vals(df):
    df_relevant = pd.DataFrame()
    for i in df.columns:
        if df[i].cov(df['num']) < -0.3 or df[i].cov(df['num']) > 0.3:
            df_relevant[i] = df[i]
    return df_relevant

def get_continuous_vals(df):
    df_continuous = pd.DataFrame()

    df_continuous.insert(0, 'age', df['age'])
    df_continuous.insert(1, 'trestbps', df['trestbps'])
    df_continuous.insert(2, 'chol', df['chol'])
    df_continuous.insert(3, 'thalach', df['thalach'])
    df_continuous.insert(4, 'oldpeak', df['oldpeak'])

    return df_continuous

def get_continuous_vals_with_num(df):
    df_continuous_with_num = get_continuous_vals(df)
    df_continuous_with_num.insert(5, 'num', df['num'])

    return df_continuous_with_num


def heatmap(df):
    resolution = 20*len(df.columns)
    print(resolution)
    g = plt.figure(figsize=(10,10),dpi=resolution)
    sns.heatmap(df.corr(), cmap = 'seismic', vmax = 1, vmin = -1,
            square = True, linewidths=1, annot=True, annot_kws={"size": 100/len(df.columns)})
    plt.tight_layout()
    return g

def full_heatmap(df):
    fig = heatmap(df)
    fig.savefig('images/Full_heatmap.png')
    plt.close(fig)

def highest_var_heatmap(df):
    usable_data = extract_relevant_vals(df)
    fig = heatmap(usable_data)
    fig.savefig('images/High_var_heatmap.png')
    plt.close(fig)

def continuous_heatmap(df):
    df = get_continuous_vals_with_num(df)
    fig = heatmap(df)
    fig.savefig('images/Continuous_heatmap.png')
    plt.close(fig)


def scatter_matrix(df):
    sns.set_theme(style="white")
    sns.pairplot(df, diag_kind='kde')
    plt.savefig('images/Scatter_matrix.png')

def scatter_with_hue(df):
    useful_cols = get_continuous_vals_with_num(df)
    sns.set_theme(style="whitegrid")
    sns.pairplot(useful_cols, diag_kind='kde', palette=sns.color_palette("hls", 5), hue='num')

    plt.savefig('images/scatter_with_hue.png')

def continuous_scatter(df):
    df = get_continuous_vals(df)
    sns.set_theme(style="white")
    sns.pairplot(df, diag_kind='kde', kind = 'reg', plot_kws={'line_kws':{'color':'red'}})
    plt.savefig('images/Continuous_scatter_matrix.png')


def continuous_scatter_with_hue(df):
    df = get_continuous_vals_with_num(df)
    useful_cols = get_continuous_vals_with_num(df)
    sns.set_theme(style="whitegrid")
    sns.pairplot(useful_cols, diag_kind='kde', palette='viridis', hue='num')

    plt.savefig('images/continuous_scatter_with_hue.png')

def boxplot(df):
    for i in df.columns:
        fig = plt.fi



        plt.close()




data = pd.read_csv('cleaned_cleveland.csv',index_col=[0])
# Firstly, we load in the data into a dataframe.

data = normalize_data(data)

boxplot(data)
