# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants to develop a machine learning model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements. This project develops a **K-Nearest Neighbors (KNN)** classification model to classify tumors accurately.

## Dataset
- **Name:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** [Kaggle — uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Records:** 569
- **Features:** 30 numerical diagnostic measurements (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, etc. — mean, standard error, and "worst" values for 10 cell nucleus characteristics)
- **Target:** `diagnosis` (M = Malignant, B = Benign)

> The dataset (`breast_cancer.csv` / `data.csv`) is **not included** in this repository. Download it from the Kaggle link above, rename it to `breast_cancer.csv`, and place it in the project root before running the notebook/script.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — train/test split, Label Encoding, feature scaling, K-Nearest Neighbors, evaluation metrics

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified the 30 numerical diagnostic features and the target variable (`diagnosis`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** — Checked for missing values, dropped the non-predictive `id` column and the empty trailing `Unnamed: 32` column present in the raw Kaggle CSV, label-encoded the target (`B`=0, `M`=1), split the data into 80% training / 20% testing sets (stratified to preserve class balance), and standardized all features with `StandardScaler` (essential for KNN since it is a distance-based algorithm).
3. **Model Development** — Trained a `KNeighborsClassifier` with `K = 5` on the scaled training data, then predicted class labels on the test set.
4. **Model Evaluation** — Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and generated a Confusion Matrix to visualize prediction performance.
5. **Conclusion** — Summarized key findings, the importance of feature scaling in KNN, and a key limitation of the algorithm.

## Results
| Metric | Value |
|---|---|
| Accuracy  | 0.9561 |
| Precision | 0.9744 |
| Recall    | 0.9048 |
| F1-Score  | 0.9383 |

**Key observations:**
1. The KNN model achieves strong classification performance on this dataset, as diagnostic measurements for malignant and benign tumors are generally well separated in feature space.
2. Precision and recall for the malignant class are especially important here, since a false negative (missing a malignant tumor) is more costly than a false positive in a healthcare context.
3. Feature scaling was essential; without standardization, features with larger numeric ranges (e.g., `area_mean`) would dominate the distance calculation and distort neighbor selection.

## Conclusion
This project used a K-Nearest Neighbors (KNN) classifier with K=5 to predict whether a breast tumor is malignant or benign based on diagnostic measurements from the Breast Cancer Wisconsin dataset. The model achieved strong performance across accuracy, precision, recall, and F1-score, showing that KNN can effectively separate malignant and benign cases using these features. Feature scaling proved essential for this algorithm: KNN classifies a point based on the distance to its nearest neighbors, so without standardizing all features to a common scale, features with larger numeric ranges would disproportionately dominate the distance calculation and distort classification results. One key limitation of KNN is that it is computationally expensive at prediction time, since it must calculate the distance from a new point to every point in the training set, making it slow to scale to very large datasets. It is also sensitive to the choice of K and to irrelevant or noisy features, both of which can degrade classification accuracy if not carefully tuned.

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/Radha-byte/Breast-Cancer-Classification-using-KNN.git
cd Breast-Cancer-Classification-using-KNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle, rename it to breast_cancer.csv, and place it in this folder

# 4. Run the notebook
jupyter notebook Assignment-4.ipynb
# OR run the script version
python Assignment-4.py
```

## Repository Structure
```
.
├── Assignment-4.ipynb    # Main notebook (all 5 tasks)
├── Assignment-4.py       # Script version of the same solution
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (this file)
└── confusion_matrix.png  # Generated evaluation plot (after running)
```
