# Oral Cancer Prediction Using Microbiome Data

<p align='center'>
<img width="800" alt="20250814_1610_Oral Cancer Insights_simple_compose_01k2mb5bswe7xvy7k147sd21j9" src="https://github.com/user-attachments/assets/a122a6bc-91ce-4a14-bfc7-4a503c579fb7" />
</p>

This project implements a machine learning pipeline for predicting oral cancer using microbiome data from **The Cancer Microbiome Atlas (TCMA)**. It features a professional, PyTorch Lightning-style CLI for flexible configuration.

---

## Overview

* **Goal:** Predict oral cancer likelihood from microbial profiles derived from 16S rRNA and WGS data.
* **Model Used:** Random Forest Classifier (class\_weight = 'balanced').
* **Explainability:** SHAP (SHapley Additive exPlanations).
* **Tools:** scikit-learn, pandas, matplotlib, shap, jsonargparse.

---

## Project Structure

```
OralCancerPrediction/
├── data/
│   ├── raw/                  # Place TCMA files here
│   └── processed/            # Outputs from preprocessing
│       ├── merged_with_labels.csv
│       └── selected_features.txt
├── oral_cancer/              # Main Python package
│   ├── __init__.py
│   ├── config.py
│   ├── downloader.py         # Data setup logic
│   ├── preprocessing.py      # Feature engineering
│   ├── modeling.py           # Model logic
│   ├── evaluation.py         # Metrics & Plotting
│   ├── data_loading.py       # DataModule
│   └── utils.py
├── scripts/                  # Execution scripts
│   └── train.py              # CLI Entry Point
├── run/                      # Shell launchers
│   └── train.sh              # Bash launcher
├── models/                   # Saved models (generated)
│   └── rf_model.pkl
├── outputs/                  # Results (images, metrics) (generated)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── ...
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

The pipeline is executed via the `run/train.sh` script, which handles environment setup and arguments.

### 1. Standard Run

Running the script without arguments will use default settings and automatically check/setup your data directory.

```bash
bash run/train.sh
```

**Note on Data:** If the required TCMA files are missing, the script will output a warning with the exact list of files you need to download manually to `data/raw/TCMA/tb09j6496/`.

### 2. Advanced Configuration (CLI)

You can override any parameter using dot-notation, similar to PyTorch Lightning:

**Override Model Hyperparameters:**

```bash
bash run/train.sh --model.n_estimators=500 --model.max_depth=10
```

**Override Data Settings:**

```bash
bash run/train.sh --data.test_size=0.2 --data.seed=123
```

**Print Configuration:**
View the full configuration tree:

```bash
bash run/train.sh --print_config
```

**Help:**
See all available options:

```bash
bash run/train.sh --help
```

---

## Model Performance

The trained model supports **non-invasive oral cancer prediction** by identifying patterns in microbiome profiles.

* **Accuracy:** 92.89%
* **AUROC:** 0.9714
* **PR-AUC:** 0.9588

### key Visualizations

| Confusion Matrix | ROC Curve |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/5cea0344-c120-402d-8074-2ae8590372e3" width="300" /> | <img src="https://github.com/user-attachments/assets/186b44e7-361d-44e8-99f5-457a57e2e55e" width="300" /> |

### SHAP Explainability

SHAP values reveal the **most influential microbial features**. Positive values (red) increase cancer probability, while negative values (blue) reduce it.

<p align="center">
  <img src="https://github.com/user-attachments/assets/53f14d6f-0f22-440c-a5d4-4e4540b5df5b" width="500" />
</p>

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
