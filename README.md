# Oral Cancer Prediction Using Microbiome Data

<p align='center'>
<img width="800" alt="20250814_1610_Oral Cancer Insights_simple_compose_01k2mb5bswe7xvy7k147sd21j9" src="https://github.com/user-attachments/assets/a122a6bc-91ce-4a14-bfc7-4a503c579fb7" />
</p>

This project implements a machine learning pipeline for predicting oral cancer using microbiome data from **The Cancer Microbiome Atlas (TCMA)**. It covers data preprocessing, feature selection, model training, evaluation, and model explainability using SHAP values.

---

## Overview

* **Goal:** Predict oral cancer likelihood from microbial profiles derived from 16S rRNA and WGS data.
* **Model Used:** Random Forest Classifier (class\_weight = 'balanced').
* **Explainability:** SHAP (SHapley Additive exPlanations).
* **Tools:** scikit-learn, pandas, matplotlib, shap, joblib.

---

## Data Source

Due to licensing restrictions, **the full TCMA dataset is not included** in this repository.

### To Reproduce:

Download the following files from TCMA:

* [`bacteria.WGS.solid.case.clr.txt`](https://tcma.pratt.duke.edu/downloads)
* `metadata.WGS.solid.case.txt`

Place them here:

```
data/raw/TCMA/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Outputs:

* `data/processed/merged_with_labels.csv`
* `data/processed/selected_features.txt`

---

## Preprocessing Pipeline

* Only TCMA data is used due to HOMD inconsistencies.
* Data merging, cleaning, imputation, scaling, and feature selection applied.
* Sequential Feature Selection (SFS) selects **17 most informative features**.
* Feature `1678.0` removed due to noise.

---

## Model Training & Evaluation

The trained model supports **non-invasive oral cancer prediction** by identifying patterns in microbiome profiles.

**Metrics:**

* **Accuracy:** 92.89%
* **AUROC:** 0.9714
* **PR-AUC:** 0.9588

### Confusion Matrix

<p align="center">
  <img src="https://github.com/user-attachments/assets/5cea0344-c120-402d-8074-2ae8590372e3" width="380" alt="confusion_matrix"/>
</p>

**Interpretation:**

* **True Negatives (TN):** 269 samples correctly classified as non-cancer.
* **True Positives (TP):** 123 samples correctly classified as cancer.
* **False Positives (FP):** 13 non-cancer samples misclassified as cancer.
* **False Negatives (FN):** 17 cancer samples missed by the model.

This reflects **high specificity** and **balanced sensitivity**.

---

### ROC Curve

<p align="center">
  <img src="https://github.com/user-attachments/assets/186b44e7-361d-44e8-99f5-457a57e2e55e" width="380" alt="roc_curve"/>
</p>

**Interpretation:**

* **AUC = 0.97**, indicating excellent discriminatory ability between cancer and non-cancer cases.
* The curve stays close to the top-left corner, suggesting **low false positive rate** and **high true positive rate**.

---

## SHAP Explainability

<p align="center">
  <img src="https://github.com/user-attachments/assets/53f14d6f-0f22-440c-a5d4-4e4540b5df5b" width="500" alt="shap_summary_plot"/>
</p>

**Interpretation:**

* SHAP interaction values reveal the **most influential microbial features** for predicting oral cancer.
* Positive SHAP values (red) indicate features that **increase cancer prediction probability**, while negative values (blue) suggest features that **reduce it**.
* This enables transparency and potential **biological insight** into cancer-related microbiome patterns.

For deeper exploration, refer to: `notebooks/_02_SHAP_Explainability.ipynb`

---

## Project Structure

```
OralCancerPrediction/
├── data/
│   ├── raw/                  # Place TCMA files here
│   └── processed/            # Outputs from preprocessing
│       ├── merged_with_labels.csv
│       └── selected_features.txt
├── notebooks/
│   ├── _00_brief.ipynb
│   ├── _01_OralCancer_Modeling.ipynb
│   ├── _02_SHAP_Explainability.ipynb
│   └── _03_Deployment_Testing.ipynb
├── models/
│   └── rf_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary_plot.png
│   ├── metrics_summary.json
│   └── metrics_summary.txt
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Running the Full Pipeline

```bash
python main.py
```

Or use the step-by-step Jupyter Notebook:

```bash
notebooks/_01_OralCancer_Modeling.ipynb
```

---

## Contributor

<div>
<table align="center">
  <tr>
        <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br />
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/madonna-mosaad" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="150px;" alt="Madonna Mosaad"/>
        <br />
        <sub><b>Madonna Mosaad</b></sub>
      </a>
    </td>
         <td align="center">
      <a href="https://github.com/Mazenmarwan023" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
        <br />
        <sub><b>Mazen Marwan</b></sub>
      </a>
    </td>    
  </tr>
</table>
</div>
