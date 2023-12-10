#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed to make results reproducable
np.random.seed(42)

np.random.seed(42)
plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")

#Mounting the google drive for the data to be read
from google.colab import drive
drive.mount('/content/drive',force_remount = True)

"""# Data Read"""

df = pd.read_csv('/content/Genetic Variant Classifications.zip',dtype={0: object, 38: str, 40: object})
df.fillna(0,inplace=True)
df.head()

# Display the first few rows of the dataset
print(df.head())

# Check column names and data types
print(df.info())

# Summary statistics for numerical columns
print(df.describe())

""" EDA Analysis"""

# Correlation heatmap for numerical columns
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Features histograms
df.drop('CLASS',axis=1).hist(figsize=(12,7))
plt.suptitle("Features histograms", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
sns.countplot(x='CLASS',data=df)
plt.title("Target label histogram")
plt.show()

# Balance
g = df.groupby('CLASS')
df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
# Extract smaller sample to avoid memory error later, when training starts
df_balanced = df_balanced.sample(1000)

f, ax = plt.subplots(1,2)
# Before balancing plot
df.CLASS.value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title("Before")
ax[0].set_xlabel("CLASS value")
ax[0].set_ylabel("Count")
# After balanced plot
df_balanced.CLASS.value_counts().plot(kind='bar',ax=ax[1])
ax[1].set_title("After")
ax[1].set_xlabel("CLASS value")
ax[1].set_ylabel("Count")

plt.suptitle("Balancing data by CLASS column value")
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

X=df_balanced.drop('CLASS',axis=1)
# One hot encoding
X=pd.get_dummies(X, drop_first=True)
y=df_balanced['CLASS']
y=pd.get_dummies(y, drop_first=True)

# Train/test split
train_X, test_X, train_y, test_y = train_test_split(X, y)

# Normalize using StandardScaler
scaler=StandardScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)

# Histogram of target labels distribution
test_y.hist()
plt.title("Target feature distribution: CLASS values")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()

numeric_df = df.select_dtypes(include=['number'])

print(numeric_df.head())

"""# Unsupervised learning algorithms

## K-Means Clustering with Numeric Columns"""

# Selecting the number of clusters (k)
k = 5

# Instantiate KMeans
kmeans = KMeans(n_clusters=k)

# Fit the model to your numeric data
kmeans.fit(numeric_df)

# Accessing cluster labels for each data point
cluster_labels = kmeans.labels_

# Visualizing clusters (example with first two features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=cluster_labels, palette='viridis')
plt.title('K-Means Clustering on Numeric Columns')
plt.xlabel('Numeric Feature 1')
plt.ylabel('Numeric Feature 2')
plt.show()

"""# Supervised learning"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

models = [
    LogisticRegressionCV()
]

results = {}
for model in models:
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    model_name = model.__class__.__name__

    # Store accuracy
    accuracy = accuracy_score(test_y, pred_y)
    results[model_name] = {'Accuracy': accuracy}

    # Generate classification report
    report = classification_report(test_y, pred_y)
    results[model_name]['Classification Report'] = report

    # Generate confusion matrix
    matrix = confusion_matrix(test_y, pred_y)
    results[model_name]['Confusion Matrix'] = matrix

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Printing results (accuracy and classification report)
for model_name, metrics in results.items():
    print(f"===== {model_name} =====")
    print(f"Accuracy: {metrics['Accuracy']}")
    print("Classification Report:")
    print(metrics['Classification Report'])
    print("=" * 30)
