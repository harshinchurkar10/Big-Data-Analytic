# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and Data analysis with Co-relation Matrix

# %%
# importing the basic library
import pandas as pd

# %%
import os

# %%
os.getcwd()

# %%
os.chdir("C:\\Users\\hp\\Desktop\\DATA SCIENCE")

# %%
data = pd.read_csv("diabetes.csv")

# %%
data.head()

# %%
data.tail()

# %%
data.info()

# %%
data.describe()

# %%
data.shape

# %%
data.size

# %%
data.ndim

# %%
data.columns

# %%
data.isna()

# %%
data.isna().any()

# %%
data.isna().sum()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# correlation
corr = data.corr()

# %%
sns.heatmap(data.corr())

# %%
plt.figure(figsize=(16, 6))
sns.heatmap(data.corr())

# %%
plt.figure(figsize=(14, 6))
sns.heatmap(data.corr(), annot=True)

# %%
