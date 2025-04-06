# %% [markdown]
# # LOGISTICS REGRESSION
# 

# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim :  To perform and analysis of Logistic Regression Algorithm

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# %%
 import os

# %%
 os.getcwd()

# %%
os.chdir("C:\\Users\\hp\\Desktop\\Big data analytics")

# %%
df = pd.read_csv("heart.csv")

# %%
df.head()

# %%
df.describe()

# %%
df.info()

# %%
df.isna().sum()

# %%
df

# %%
df.isna().sum()

# %%
 #Splitting the dependent and independent variables.
 x = df.drop("target",axis=1)
 y = df['target']

# %%
 x #checking the features

# %% [markdown]
# # Train Test Split
# 

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# %%
 y_train

# %%


# %% [markdown]
# # Logistic Regression Algorithm
# 

# %%
 from sklearn.linear_model import LogisticRegression
 model = LogisticRegression().fit(x_train,y_train)
 model.score(x_train, y_train)

# %%



