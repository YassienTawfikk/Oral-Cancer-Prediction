"""
NOTE:
-----
Please make sure to run the preprocessing step before executing this script.

Run the following command first:
    python src/_00_preprocessing.py

This will generate the file:
    data/processed/TCMA/merged_with_labels.csv

The pipeline below depends on that file for training and evaluation.
"""

import os
from src._01_fetching import load_data, split_data
from src._02_modeling import train_random_forest, get_shap_values
from src._03_evaluation import (
    evaluate_model,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curve
)
from src._04_utils import save_model, save_json
import matplotlib.pyplot as plt
import shap


def main():
    #  Ensure required output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    X, y = load_data("data/processed/TCMA/merged_with_labels.csv")

    # 2. Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train Random Forest
    model = train_random_forest(X_train, y_train)

    # 4. Evaluate model
    y_pred, y_proba, metrics = evaluate_model(model, X_test, y_test, '../outputs/metrics_summary.txt')

    # 5. Save performance metrics
    save_json(metrics, "outputs/metrics_summary.json")

    # 6. Save confusion matrix and ROC
    plot_and_save_confusion_matrix(y_test, y_pred, "outputs/confusion_matrix.png")
    plot_and_save_roc_curve(model, X_test, y_test, "outputs/roc_curve.png")

    # 7. SHAP explainability
    print("Generating SHAP summary plot...")
    shap_values = get_shap_values(model, X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot – Class 1 (Cancer)")
    plt.savefig("outputs/shap_summary_plot.png", bbox_inches='tight')
    plt.close()

    # 8. Save model and metadata
    save_model(model, "models/rf_model.pkl")
    model_info = {
        "model_type": "RandomForestClassifier",
        "date": "2025-05-11",
        "features_used": list(X.columns),
        "n_features": X.shape[1],
        "test_samples": X_test.shape[0],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"]
    }
    save_json(model_info, "models/best_model_metadata.json")

    print("Pipeline complete ✅")


if __name__ == "__main__":
    main()
