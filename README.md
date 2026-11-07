# Machine Learning — Assignment 2

Name: Thoolika CA
BITS ID: 2026AK05165
Programme: M.Tech Artificial Intelligence and Machine Learning
Institution: BITS Pilani — WILP

## a. Problem statement

Classify breast cancer samples as malignant or benign using numerical
measurements of cell nuclei. Implement five classification models on
the same dataset and compare their performance using Accuracy, AUC,
Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

Provide a Streamlit application that loads saved models and evaluates
labelled test CSV files without retraining.

## b. Dataset description

Dataset: Breast Cancer Wisconsin (Diagnostic)
Public repository: UCI Machine Learning Repository
Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

The dataset was loaded using scikit-learn's load_breast_cancer function.

- Instances: 569
- Input features: 30 numerical features
- Target column: target
- Target encoding: 0 = malignant, 1 = benign
- Malignant samples: 212
- Benign samples: 357
- Missing values: 0
- Duplicate rows: 0

The features describe radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry and fractal dimension.
Each measurement is represented by its mean, standard error and
worst-value summary, giving 30 input features.

### Data preparation

The dataset was split into 455 training samples (80%) and 114 test
samples (20%), using random_state=42 and stratify=y.

All five models use the same training and test rows. The target column
is excluded from the input features.

Logistic Regression, KNN and SVM use StandardScaler within pipelines.
Scaling parameters are learned from the training data only.
Decision Tree and Gaussian Naive Bayes use the original feature values.

The saved test_data.csv contains 114 rows, all 30 input features and
the target column.

## c. GitHub Repository Link

https://github.com/Thoolikaca/ml-assignment-2-thoolika

## d. Models used

1. Logistic Regression: StandardScaler and max_iter=2000.
2. Decision Tree Classifier: default settings with random_state=42.
3. K-Nearest Neighbour: StandardScaler and n_neighbors=5.
4. Gaussian Naive Bayes: default settings.
5. Support Vector Machine: StandardScaler, RBF kernel,
   probability=True and random_state=42.

### Evaluation convention

All results below are calculated on the held-out test set.
Malignant (label 0) is the positive class for Precision, Recall and F1.
AUC uses the predicted probability of the malignant class.
MCC uses the original target labels and predicted labels.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.8478 | 0.9286 | 0.8864 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9268 | 0.9048 | 0.9157 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 112 of 114 predictions were correct. It has the highest AUC.|
| Decision Tree | 104 predictions were correct. It has the lowest accuracy and precision.|
| KNN | 109 predictions were correct. Its accuracy is between Decision Tree and Logistic Regression.|
| Naive Bayes | 107 predictions were correct. Its AUC is higher than KNN’s, but its recall is lower. |
| Support Vector Machine | 112 predictions were correct. It ties Logistic Regression on five metrics, with a slightly lower AUC. |
| Overall, winner for this dataset? | Logistic Regression is the overall winner for this test split. It achieved 98.25% accuracy, equal to SVM, but had a slightly higher AUC of 0.9954 compared with SVM’s 0.9950. The other four metrics were also equal to SVM. |

These observations apply to this particular train/test split.

## Running the application

Use Python 3.12, matching the BITS Virtual Lab environment.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

### Using the application

1. Open the application in the browser.
2. Select a classifier from the sidebar.
3. Use the supplied test dataset or upload a labelled test CSV.
4. View the six evaluation metrics and the five-model comparison.
5. View the selected model's confusion matrix and classification report.

The uploaded CSV must contain all 30 feature columns listed in
metadata.json and the target column. Target values must be
0 (malignant) or 1 (benign).

The application loads saved models from the model folder.
It does not retrain models on uploaded test data.

## Project files

- app.py: Streamlit application.
- model_training.ipynb: Dataset preparation, training and evaluation source.
- requirements.txt: Required Python package versions.
- README.md: Project description, results and instructions.
- test_data.csv: Held-out test features and target labels.
- model/: Five saved classifiers, including scaling within pipelines where used.
- metadata.json: Feature names, target labels and model filenames.
- model_comparison.csv: Evaluation results for all five models.
Performance on other samples or splits may differ.