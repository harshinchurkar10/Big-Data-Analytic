# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To Find Unique and Duplicates Value Count in given dataset

# %%
import pandas as pd

# %%
import os

# %%
os.getcwd()

# %%
os.chdir("C:\\Users\\hp\\Desktop\\DATA SCIENCE")

# %%
df = pd.read_csv("diabetes.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.size

# %%
df.shape

# %%
df.ndim

# %%
df.info()

# %%
df.describe()

# %%
df.columns

# %%
df.isna()

# %%
df.isna().any()

# %%
df.isna().sum()

# %%
df["Age"].unique()

# %%
df["Age"].duplicated()

# %%
df["Age"].duplicated().sum()

# %%


# %%
