# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and analysis for Normal Distribution in given dataset

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
df.info()

# %%
df.describe()

# %%
df.size

# %%
df.shape

# %%
df.ndim

# %%
df.columns

# %%
df.isna()

# %%
df.isna().any()

# %%
df.isna().sum()

# %%


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.distplot(df, bins=20)
plt.show()

# %%
sns.distplot(df["Glucose"], bins=20)
plt.show()

# %%
sns.distplot(df["Age"], bins=20)
plt.show()

# %%
sns.distplot(df["BloodPressure"], bins=20)
plt.show()

# %%
sns.distplot(df["SkinThickness"], bins=20)
plt.show()

# %%
import matplotlib.pyplot as plt

# %%
plt.hist(df["Age"], bins=30, color="blue", edgecolor="black", alpha=0.7)

# %%
