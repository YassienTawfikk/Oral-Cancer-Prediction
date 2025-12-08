"""
Oral Cancer Prediction CLI
==========================
Uses jsonargparse to provide a Lightning-style CLI.
"""

import os
import sys
import logging
import shap
import matplotlib.pyplot as plt
from jsonargparse import CLI

# Ensure package path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oral_cancer.data_loading import OralCancerDataModule
from oral_cancer.modeling import OralCancerModel
from oral_cancer.downloader import setup_data_directory
from oral_cancer.evaluation import (
    evaluate_model,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curve,
    plot_heatmap,
    plot_pipeline
)
from oral_cancer.utils import save_json
from oral_cancer.config import OUTPUTS_DIR, MODELS_DIR

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(data: OralCancerDataModule, model: OralCancerModel):
    """
    Main pipeline function.
    
    Args:
        data: Data configuration.
        model: Model configuration.
    """
    # 1. Setup
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # 0. Download/Check Data
    setup_data_directory()
    
    logging.info("Setting up data...")
    try:
        data.setup()
    except Exception as e:
        logging.error(f"Data setup failed: {e}")
        return

    X_train, y_train = data.train_data()
    X_test, y_test = data.test_data()
    
    logging.info(f"Training with {model.n_estimators} trees...")
    
    # 2. Train
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    metrics_path = os.path.join(OUTPUTS_DIR, 'metrics_summary.txt')
    y_pred, y_proba, metrics = evaluate_model(model.model, X_test, y_test, metrics_path)
    
    logging.info(f"Results -> Accuracy: {metrics['accuracy']}, ROC AUC: {metrics['roc_auc']}")
    save_json(metrics, os.path.join(OUTPUTS_DIR, "metrics_summary.json"))

    # 4. Plots
    plot_and_save_confusion_matrix(y_test, y_pred, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
    plot_and_save_roc_curve(model.model, X_test, y_test, os.path.join(OUTPUTS_DIR, "roc_curve.png"))
    
    if hasattr(X_test, "columns"):
        taxa = list(X_test.columns)
    else:
        taxa = [f"Feature {i}" for i in range(X_test.shape[1])]
    plot_heatmap(model.model, X_test, taxa, os.path.join(OUTPUTS_DIR, "shap_heatmap.png"))
    
    # 5. Save Model
    model.save(os.path.join(MODELS_DIR, "rf_model.pkl"))
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    CLI(run_pipeline)
