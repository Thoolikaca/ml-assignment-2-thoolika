from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="Thoolika's Classification Explorer",
    layout="wide"
)

st.title("Breast Cancer Classification Explorer")
st.caption("Machine Learning Assignment 2 | Thoolika CA")

st.write(
    "Compare five trained classifiers using a labelled test dataset. "
    "This application demonstrates model evaluation for coursework."
)

# Find files relative to app.py
PROJECT_FOLDER = Path(__file__).resolve().parent

metadata = json.loads(
    (PROJECT_FOLDER / "metadata.json").read_text(encoding="utf-8")
)

feature_names = metadata["feature_names"]
target_column = metadata["target_column"]


@st.cache_resource
def load_saved_models():
    models = {}
    for model_name, filename in metadata["model_files"].items():
        models[model_name] = joblib.load(
            PROJECT_FOLDER / "model" / filename
        )
    return models


models = load_saved_models()

selected_model_name = st.sidebar.selectbox(
    "Choose a classifier",
    list(models.keys())
)

st.sidebar.info(
    "Target labels: 0 = malignant, 1 = benign. "
    "Malignant is the positive class for Precision, Recall, F1 and AUC."
)

st.subheader("1. Choose test data")

st.download_button(
    label="Download sample test CSV",
    data=(PROJECT_FOLDER / "test_data.csv").read_bytes(),
    file_name="test_data.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload a test CSV containing all 30 features and the target column",
    type=["csv"]
)

try:
    if uploaded_file is None:
        evaluation_data = pd.read_csv(PROJECT_FOLDER / "test_data.csv")
        st.info("Currently evaluating the saved held-out test dataset.")
    else:
        evaluation_data = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV loaded.")
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

# Validate the test data before making predictions
required_columns = feature_names + [target_column]
missing_columns = [
    column for column in required_columns
    if column not in evaluation_data.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if evaluation_data.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

try:
    numeric_data = evaluation_data[required_columns].apply(
        pd.to_numeric, errors="raise"
    )
except (ValueError, TypeError):
    st.error("All feature values and target labels must be numeric.")
    st.stop()

if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
    st.error("The data contains missing or infinite values.")
    st.stop()

if not numeric_data[target_column].isin([0, 1]).all():
    st.error("Target labels must be 0 (malignant) or 1 (benign).")
    st.stop()

app_X = numeric_data[feature_names]
app_y = numeric_data[target_column].astype(int)

with st.expander("Preview test data and required columns"):
    st.dataframe(numeric_data.head(10))
    st.write("Required feature columns:", feature_names)

st.write(f"Evaluating {len(app_X)} test samples.")

if app_y.nunique() < 2:
    st.warning(
        "This test data contains only one class. "
        "AUC is undefined and will appear as N/A."
    )


# Evaluate saved models without retraining
def calculate_metrics(model_name, model):
    predictions = model.predict(app_X)
    malignant_column = list(model.classes_).index(0)
    probabilities = model.predict_proba(app_X)[:, malignant_column]

    auc = (
        roc_auc_score((app_y == 0).astype(int), probabilities)
        if app_y.nunique() == 2 else np.nan
    )

    return {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(app_y, predictions),
        "AUC": auc,
        "Precision": precision_score(
            app_y, predictions, pos_label=0, zero_division=0
        ),
        "Recall": recall_score(
            app_y, predictions, pos_label=0, zero_division=0
        ),
        "F1": f1_score(
            app_y, predictions, pos_label=0, zero_division=0
        ),
        "MCC": matthews_corrcoef(app_y, predictions)
    }


app_results = pd.DataFrame([
    calculate_metrics(name, model)
    for name, model in models.items()
])

st.subheader("2. Selected model evaluation")
st.write(selected_model_name)

selected_result = app_results[
    app_results["ML Model Name"] == selected_model_name
].iloc[0]

metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_names):
    value = selected_result[metric_name]
    metric_columns[index % 3].metric(
        metric_name,
        "N/A" if pd.isna(value) else f"{value:.4f}"
    )

st.subheader("3. Compare all five models")
st.dataframe(
    app_results.style.format(
        {name: "{:.4f}" for name in metric_names},
        na_rep="N/A"
    )
)

st.subheader("4. Classification results")
selected_model = models[selected_model_name]
selected_predictions = selected_model.predict(app_X)

matrix = confusion_matrix(
    app_y, selected_predictions, labels=[0, 1]
)

st.write("Confusion matrix: rows = actual, columns = predicted")
st.dataframe(pd.DataFrame(
    matrix,
    index=["Actual malignant", "Actual benign"],
    columns=["Predicted malignant", "Predicted benign"]
))

st.write("Classification report")
report = classification_report(
    app_y,
    selected_predictions,
    labels=[0, 1],
    target_names=["Malignant", "Benign"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4))