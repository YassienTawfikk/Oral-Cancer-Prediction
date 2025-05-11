import matplotlib.pyplot as plt
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


def save_text_report(accuracy, roc_auc, pr_auc, report_text, path: str):
    """Save a human-readable text summary of the evaluation metrics."""
    with open(path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
