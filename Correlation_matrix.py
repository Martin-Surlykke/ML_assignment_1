import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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



def bar_chart_matrix(df):
    df_for_bar_chart = df[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'num']]

    print(df['thal'].describe())

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))

    label_mappings = {
        'sex': {0.0: "Female", 1.0: "Male"},
        'cp': {0.0: "No chest pain", 1.0: "Typical angina", 2.0: "Atypical angina", 3.0: "Non-anginal pain"},
        'fbs': {0.0: "Fasting blood sugar <= 120", 1.0: "Fasting blood sugar > 120"},
        'restecg': {0.0: "Normal", 1.0: "ST-T wave abnormality", 2.0: "Left ventricular hypertrophy"},
        'exang': {0.0: "No exercise induced angina", 1.0: "Exercise induced angina"},
        'slope': {0.0: "Upward slope", 1.0: "Flat slope", 2.0: "Downward slope"},
        'thal': {0.0: "Normal", 1.0: "Fixed defect", 2.0: "Reversible defect"},
        'num': {0.0: "No heart disease", 1.0: "Low-level Heart disease",
                2.0: "Mid-level heart disease", 3.0: "High level heart disease", 4.0: "Severe heart disease"}
    }
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop over each column and create a bar plot in each subplot
    for i, column in enumerate(df_for_bar_chart.columns):
        sns.countplot(x=column, data=df_for_bar_chart, ax=axes[i])

        if column in label_mappings:
            axes[i].set_xticklabels([label_mappings[column].get(val, val) for val in axes[i].get_xticks()],
                                    rotation=45, size=12, horizontalalignment='right')

        axes[i].set_title(f'Bar Chart for {column}')


data = pd.read_csv('cleaned_cleveland.csv')
# Firstly, we load in the data into a dataframe.

bar_chart_matrix(data)

