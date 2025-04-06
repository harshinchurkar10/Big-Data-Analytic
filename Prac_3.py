# %%
# Name : Harsh Inchurkar
# Roll no. : 46
# Section : 3A

# %%
# Aim : To perform and analysis of Naive Bayes, confusion matrix, K fold Cross Validation

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# # Data acquisitionuing Pandas
#

# %%
import os

# %%
os.getcwd()

# %%
os.chdir("C:\\Users\\hp\\Desktop\\Big data analytics")

# %%
data = pd.read_csv("heart.csv")

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

# %% [markdown]
# # Data preprocessing _ data cleaning _ missing value treatment
#

# %%
# check Missing Value by record

data.isna()

# %%
data.isna().any()

# %%
data.isna().sum()

# %% [markdown]
# # Removing duplicates
#

# %%
data_dup = data.duplicated().any()

# %%
data_dup

# %%
data = data.drop_duplicates()

# %%
data_dup = data.duplicated().any()

# %%
data_dup

# %% [markdown]
# # Splitting of DataSet into train and Test
#

# %%
x = data.drop("target", axis=1)
y = data["target"]

# %%
# splitting the data into training and testing data sets
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
# # Naive Bayes classifier
#

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# %%
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

# %%
y_pred = nb_classifier.predict(x_test)

# %%
accuracy_score(y_test, y_pred)

# %% [markdown]
# # confusion matrix
#

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Reds", linewidths=1, linecolor="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %%
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

# %%
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision
precision = precision_score(y_test, y_pred, average="weighted")
print(f"Precision: {precision:.4f}")

# Recall
recall = recall_score(y_test, y_pred, average="weighted")
print(f"Recall: {recall:.4f}")

# Error Rate
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%


# %% [markdown]
# # K fold Cross Validation
#

# %%
from sklearn.model_selection import KFold, cross_val_score

# %%
# Define K-Fold Cross Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform Cross Validation
scores = cross_val_score(nb_classifier, x, y, cv=kf, scoring="accuracy")

# Print results
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")

# %%
