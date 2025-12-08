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
from typing import Tuple, Dict, Any, List, Optional, Union
import shap

def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    report_txt_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Predict and compute core metrics, save report.txt (optional).
    
    Returns:
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

def save_text_report(accuracy: float, roc_auc: float, pr_auc: float, report_text: str, path: str) -> None:
    """Save a human-readable text summary of the evaluation metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)

def plot_and_save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, path: str) -> None:
    """Plot and save the confusion matrix as an image."""
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

def plot_and_save_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series, path: str) -> None:
    """Plot and save the ROC curve as an image."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(path)
    plt.close()

def plot_pipeline(path: str) -> None:
    """Draw and save a visual representation of the pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Define pipeline stages
    stages = [
        {"name": "1. Data Preprocessing",
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

def get_shap_values_local(model: Any, X: pd.DataFrame) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Compute SHAP values for a tree model. Helper for heatmap.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # If list (multiclass), return second array for class 1
    if isinstance(shap_values, list) and len(shap_values) > 1:
        return shap_values[1]
    return shap_values

def plot_heatmap(model: Any, X: pd.DataFrame, feature_names: List[str], path: str, top_n: int = 16) -> None:
    """
    Plot a heatmap of top N features by mean absolute SHAP value.
    """
    # Get SHAP values
    shap_values = get_shap_values_local(model, X)
    
    # Calculate mean absolute SHAP values
    if isinstance(shap_values, list): # Should be handled by get_shap_values_local but check safety
         # Multiclass case, ideally we want class 1
         shap_values = shap_values[1]

    # For binary classification or regression, shap_values is (n_samples, n_features)
    # or (n_samples, n_features, n_output) depending on version/model.
    # Assuming (n_samples, n_features) for binary class 1 from get_shap_values_local
    
    if len(np.shape(shap_values)) == 3:
         # Some versions return (samples, features, classes)
         mean_abs_shap = np.mean(np.abs(shap_values[:, :, 1]), axis=0)
    else:
         mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Create DataFrame
    # Ensure feature_names matches X columns if X is DataFrame, otherwise use passed names
    if hasattr(X, "columns"):
        feats = X.columns
    else:
        feats = feature_names

    mean_shap = pd.DataFrame({
        'Feature': feats,
        'SHAP Importance': mean_abs_shap
    })

    # Sort and select top features
    mean_shap = mean_shap.sort_values('SHAP Importance', ascending=False).head(top_n)

    # Create heatmap data (transposed for better visualization)
    heatmap_data = mean_shap.set_index('Feature').T

    # Plot
    plt.figure(figsize=(10, 2))  # Adjust size
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean |SHAP Value|"},
        square=True
    )
    plt.title(f"Heatmap of Top {top_n} Features by SHAP Importance", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
