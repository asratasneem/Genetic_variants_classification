# Genomic Variant Classification and Clustering Pipeline

## Overview
This project implements a machine learning pipeline for analyzing and classifying genomic variant data. The workflow integrates exploratory data analysis (EDA), class balancing, unsupervised clustering, and supervised learning.

## Objectives
- Perform exploratory data analysis on genomic variant datasets
- Visualize feature distributions and correlations
- Address class imbalance using downsampling
- Apply unsupervised clustering to explore latent structure
- Train and evaluate supervised classification models

---

## Data Processing & EDA

- Dataset loaded using Pandas
- Missing values handled using fillna()
- Correlation heatmap generated for numerical features
- Feature histograms plotted
- Target class distribution visualized

### Class Balancing
To address class imbalance:
- Downsampling applied using group-wise sampling
- Balanced dataset subset extracted for modeling

---

## Feature Engineering

- One-hot encoding applied to categorical features
- Train-test split performed
- Feature scaling using StandardScaler

---

## Unsupervised Learning

### K-Means Clustering
- Applied on numeric features
- Cluster labels generated
- 2D scatter plot visualization of clusters

---

## Supervised Learning

### Logistic Regression (Cross-Validated)
- Model: LogisticRegressionCV
- Evaluation Metrics:
  - Accuracy
  - Classification Report
  - Confusion Matrix

Confusion matrices are visualized using Seaborn heatmaps.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Reproducibility

Install required packages:

```bash
pip install -r requirements.txt
