# %% [markdown]
# # LINEAR REGRATION
#

# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and analysis of Linear Regression Algorithm

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
import os

# %%
os.getcwd()

# %%
os.chdir("C:\\Users\\hp\\Desktop\\Big data analytics")

# %%
df = pd.read_csv("heart.csv")

# %%
df

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.size

# %%
df.ndim

# %%
df.info()

# %%
df.describe()

# %%
df.isna()

# %%
df.isna().any()

# %%
df.isna().sum()

# %%
# Assiging values in X & Y
# X = df['Select all rows and attribute except the last attribute']
# y = df['target']

# method of assign
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# %%
print(X)

# %%
print(y)

# %%
# Splitting testdata into X_train,X_test,y_train,y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
print(X_train)

# %%
print(X_test)

# %%
print(y_train)

# %%
print(y_test)

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# %%
# Assigning Coefficient (slope) to m
m = lr.coef_

# %%
print("Coefficient  :", m)

# %%
# Assigning Y-intercept to a
c = lr.intercept_

# %%
print("Intercept : ", c)

# %%
lr.score(X_test, y_test) * 100

# %%


# %%


# %%


# %%
