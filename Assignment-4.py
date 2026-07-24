"""
Assignment 4 - Breast Cancer Classification using K-Nearest Neighbors (KNN)
"""

# =========================================================
# Task 1: Data Understanding
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

# 1. Load the dataset
df = pd.read_csv("breast_cancer.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and the target variable
# 'id' is an identifier (not predictive) and 'Unnamed: 32' is an empty trailing
# column present in the raw Kaggle CSV; both are dropped in preprocessing.
numerical_features = [col for col in df.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
target_variable = 'diagnosis'

print("\nNumber of numerical features:", len(numerical_features))
print("Numerical features:", numerical_features)
print("Target variable:", target_variable)

# 4. Dataset information and summary statistics
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nDataset shape:", df.shape)

# =========================================================
# Task 2: Data Preprocessing
# =========================================================

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove unnecessary columns: 'id' (identifier) and 'Unnamed: 32' (empty column)
cols_to_drop = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

print(f"\nDropped columns: {cols_to_drop}")
print("Shape after dropping unnecessary columns:", df.shape)

# Encode the target variable (Malignant/Benign -> 1/0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # B=0, M=1
print("\nTarget classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# Select features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Normalize/standardize feature values
# KNN relies on distance calculations, so features must be on the same scale;
# without this, large-magnitude features (e.g., area) would dominate the distance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# Task 3: Model Development
# =========================================================

# 1 & 2. Train a K-Nearest Neighbors classifier with K = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 3. Predict the class labels for the test dataset
y_pred = knn.predict(X_test_scaled)

print("\nModel trained successfully with K = 5.")

# =========================================================
# Task 4: Model Evaluation
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix - Breast Cancer Classification (KNN, K=5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Observations
print("""
Observations:
1. The KNN model achieves strong classification performance on this dataset,
   as diagnostic measurements for malignant and benign tumors are generally
   well separated in feature space.
2. Precision and recall for the malignant class are especially important here,
   since a false negative (missing a malignant tumor) is more costly than a
   false positive in a healthcare context.
3. Feature scaling was essential; without standardization, features with larger
   numeric ranges (e.g., area_mean) would dominate the distance calculation and
   distort neighbor selection.
""")

# =========================================================
# Task 5: Conclusion
# =========================================================
conclusion = """
This project used a K-Nearest Neighbors (KNN) classifier with K=5 to predict
whether a breast tumor is malignant or benign based on diagnostic measurements
from the Breast Cancer Wisconsin dataset. The model achieved strong performance
across accuracy, precision, recall, and F1-score, showing that KNN can
effectively separate malignant and benign cases using these features. Feature
scaling proved essential for this algorithm: KNN classifies a point based on
the distance to its nearest neighbors, so without standardizing all features to
a common scale, features with larger numeric ranges would disproportionately
dominate the distance calculation and distort classification results. One key
limitation of KNN is that it is computationally expensive at prediction time,
since it must calculate the distance from a new point to every point in the
training set, making it slow to scale to very large datasets. It is also
sensitive to the choice of K and to irrelevant or noisy features, both of
which can degrade classification accuracy if not carefully tuned.
"""
print(conclusion)
