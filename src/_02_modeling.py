from sklearn.ensemble import RandomForestClassifier
import shap


def train_random_forest(
        X_train, y_train,
        n_estimators: int = 100,
        class_weight: str = 'balanced',
        random_state: int = 42
):
    """
    Train and return a RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def get_shap_values(model, X):
    """
    Compute SHAP values for a tree model; return values for positive class if applicable.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # If list (multiclass), return second array for class 1
    if isinstance(shap_values, list) and len(shap_values) > 1:
        return shap_values[1]
    return shap_values


# src/_03_evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def evaluate_model(model, X_test, y_test):
    """
    Predict and compute core metrics and classification report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "classification_report": report
    }
    return y_pred, y_proba, metrics


def plot_and_save_confusion_matrix(y_test, y_pred, path: str):
    """
    Plot and save confusion matrix.
    """
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


def plot_and_save_roc_curve(model, X_test, y_test, path: str):
    """
    Plot and save ROC curve.
    """
    disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(path)
    plt.close()
