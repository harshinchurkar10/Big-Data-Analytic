# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and Data analysis with Confusion matrix

# %%
import pandas as pd
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
df.head()

# %%
df.tail()

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

# %% [markdown]
# # Splitting of DataSet into train and Test
#

# %%
x = df.drop("target", axis=1)
y = df["target"]

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# %%
x_train

# %%
x_test

# %%
y_train

# %%
y_test

# %% [markdown]
# # Logistic Regression
#

# %%
df.head()

# %%
from sklearn.linear_model import LogisticRegression

# %%
log = LogisticRegression()
log.fit(x_train, y_train)

# %%
y_pred1 = log.predict(x_test)

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test, y_pred1) * 100

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
cm = confusion_matrix(y_test, y_pred1)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges", linewidths=1, linecolor="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %%
