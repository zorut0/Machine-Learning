#a

import pandas as pd
from sklearn.datasets import load_iris

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print(df)

df_csv = pd.read_csv('students.csv')
print(df_csv.head())

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())

#b

from scipy import stats
import numpy as np

data = {'Scores': [89, 78, 90, 92, 85, 78, 90]}
df_stats = pd.DataFrame(data)
print(df_stats)

mean_val = df_stats['Scores'].mean()
median_val = df_stats['Scores'].median()
mode_val = stats.mode(df_stats['Scores'], keepdims=True)[0][0]
variance_val = df_stats['Scores'].var()
std_dev_val = df_stats['Scores'].std()

print(mean_val)
print(median_val)
print(mode_val)
print(variance_val)
print(std_dev_val)

#c

import numpy as np

data = {
    'Name': ['Anna', 'Ben', 'Clara', 'Diana'],
    'Age': [23, 25, np.nan, 30],
    'Score': [88, 92, 85, np.nan]
}
df_pre = pd.DataFrame(data)

reshaped = df_pre.values.reshape(-1, 2)

filtered = df_pre[df_pre['Age'] > 24]

data2 = {
    'Name': ['Anna', 'Ben', 'Clara', 'Diana'],
    'Department': ['IT', 'HR', 'Finance', 'Marketing']
}
df2 = pd.DataFrame(data2)

merged = pd.merge(df_pre, df2, on='Name')

df_pre['Age'] = df_pre['Age'].fillna(df_pre['Age'].mean())
df_pre['Score'] = df_pre['Score'].fillna(df_pre['Score'].mean())

df_pre['Score_norm'] = (df_pre['Score'] - df_pre['Score'].min()) / (df_pre['Score'].max() - df_pre['Score'].min())

print(reshaped)
print(filtered)
print(merged)
print(df_pre)
