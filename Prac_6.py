# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and analysis of ANOVA parametric Test

# %% [markdown]
# # One Way F-test(Anova) :-
#

# %% [markdown]
# It tell whether two or more groups are similar or not based on their mean similarity and f-score.
#
# Example : there are 3 different category of iris flowers and their petal width and need to check whether all 3 group are similar or not
#

# %%
import seaborn as sns

df1 = sns.load_dataset("iris")

# %%
df1.head()

# %%
df1.tail()

# %%
df_anova = df1[["petal_width", "species"]]

# %%
import pandas as pd

grps = pd.unique(df_anova.species.values)

# %%
grps

# %%
d_data = {grp: df_anova["petal_width"][df_anova.species == grp] for grp in grps}

# %%


# %%
d_data

# %%
import scipy.stats as stats

# %%
F, p = stats.f_oneway(d_data["setosa"], d_data["versicolor"], d_data["virginica"])

# %%
print(p)

# %%
if p < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

# %%
