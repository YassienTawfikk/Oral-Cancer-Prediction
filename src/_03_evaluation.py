import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def evaluate_model(model, X_test, y_test, report_txt_path=None):
    """
    Predict and compute core metrics, save report.txt (optional), and return:
      - y_pred: class predictions
      - y_proba: predicted probabilities for class 1
      - metrics: dict with accuracy, ROC AUC, PR AUC, and classification report
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    # Save report as .txt if path is provided
    if report_txt_path:
        save_text_report(accuracy, roc_auc, pr_auc, report_text, report_txt_path)

    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "classification_report": report_dict
    }
    return y_pred, y_proba, metrics


def plot_and_save_confusion_matrix(y_test, y_pred, path: str):
    """Plot and save the confusion matrix as an image."""
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


def plot_and_save_roc_curve(model, X_test, y_test, path: str):
    """Plot and save the ROC curve as an image."""
    disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(path)
    plt.close()





def plot_pipeline(path: str):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Define pipeline stages
    stages = [
        {"name": "1. data Preprocessing",
         "steps": ["Merge raw tables", "Clean/impute", "CLR scaling", "Feature selection (SFS)"]},
        {"name": "2. Model Training", "steps": ["RandomForest", "Class balancing", "Hyperparameter tuning"]},
        {"name": "3. Evaluation", "steps": ["AUROC/PR-AUC", "SHAP analysis", "Confusion matrix"]}
    ]

    # Draw stages
    for i, stage in enumerate(stages):
        x = i * 3.5
        ax.add_patch(Rectangle((x, 0), 3, 5, fill=False, edgecolor="black", lw=2))
        ax.text(x + 1.5, 4.5, stage["name"], ha="center", va="center", fontsize=12, weight="bold")

        # Draw steps
        for j, step in enumerate(stage["steps"]):
            ax.text(x + 1.5, 3.5 - j, step, ha="center", va="center", fontsize=10)

    # Arrows
    for i in range(2):
        ax.arrow(i * 3.5 + 3, 2.5, 0.5, 0, head_width=0.2, head_length=0.2, fc="k")

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    plt.title("End-to-End Pipeline for Oral Cancer Prediction", pad=20, fontsize=14)
    plt.savefig(path)
    plt.close()


import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_shap_values(model, X):
    """
    Compute SHAP values for a tree model; return values for positive class if applicable.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # If list (multiclass), return second array for class 1
    if isinstance(shap_values, list) and len(shap_values) > 1:
        return shap_values[1]
    print(shap_values)
    return shap_values


def plot_heatmap(model, X, feature_names, path: str, top_n=16):
    """
    Plot a heatmap of top N features by mean absolute SHAP value

    Parameters:
    - model: Your trained model
    - X: Input data (features) to explain
    - feature_names: List of feature names (taxa names)
    - path: Where to save the heatmap
    - top_n: Number of top features to show
    """
    # Get SHAP values
    shap_values = get_shap_values(model, X)
    print(f"SHAP values shape: {np.shape(shap_values)}")

    # For binary classification, we get two sets of SHAP values (one per class)
    # We'll use the values for class 1 (positive class) which is what get_shap_values returns
    if len(shap_values.shape) == 3:
        # If we have multiple outputs, take mean across samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        # For binary case, we should have shape (n_features, 2)
        # We'll take the values for the positive class (index 1)
        if mean_abs_shap.shape[1] == 2:
            mean_abs_shap = mean_abs_shap[:, 1]
        else:
            mean_abs_shap = mean_abs_shap.mean(axis=1)  # average across outputs if >2 classes
    else:
        # For single output, just take mean absolute across samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Create DataFrame
    mean_shap = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': mean_abs_shap
    })

    # Sort and select top features
    mean_shap = mean_shap.sort_values('SHAP Importance', ascending=False).head(top_n)

    # Create heatmap data (transposed for better visualization)
    heatmap_data = mean_shap.set_index('Feature').T

    # Plot
    plt.figure(figsize=(10, 1))  # Adjust size based on number of features
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean |SHAP Value| (Impact on Prediction)"},
        square=True
    )
    plt.title(f"Heatmap of Top {top_n} Microbiome Taxa by SHAP Importance", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([])  # Remove the y-axis label since we're showing one row
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_text_report(accuracy, roc_auc, pr_auc, report_text, path: str):
    """Save a human-readable text summary of the evaluation metrics."""

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
