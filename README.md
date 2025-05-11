# Oral Cancer Prediction Using Microbiome Data

This project builds a machine learning pipeline to predict oral cancer using microbiome data sourced from The Cancer Microbiome Atlas (TCMA). It includes end-to-end preprocessing, feature selection, model training, evaluation, and explainability using SHAP values.

---

## 📌 Overview

* **Goal:** Predict oral cancer based on microbial features derived from 16S rRNA and WGS data.
* **Model Used:** Random Forest Classifier
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Tools:** scikit-learn, pandas, matplotlib, shap, joblib

---

## 🧬 Data Source

Due to data licensing and privacy considerations, **the full TCMA dataset is not included** in this repository.

### 📥 To Reproduce:

Please download the following data files from TCMA:

* [`bacteria.WGS.solid.case.clr.txt`](https://tcma.pratt.duke.edu/downloads)
* `metadata.WGS.solid.case.txt`

> 📂 Place them in the following directory:

```
data/raw/TCMA/
```

You must also download and install the required Python packages using:

```bash
pip install -r requirements.txt
```

Then, run the preprocessing script as described below.

---

## 🧪 Preprocessing Pipeline

* We use **only TCMA** (not HOMD) due to data inconsistency issues.
* Merging, cleaning, imputing, scaling, and feature selection are performed.
* Sequential Feature Selection (SFS) chooses the most informative 17 features.
* Feature `1678.0` is explicitly dropped due to noise.

### ⚙️ Run preprocessing (takes time depending on CPU):

```bash
python src/preprocessing.py
```

This will generate:

* `data/processed/merged_with_labels.csv`
* `data/processed/selected_features.txt`

---

## 🤖 Model Training & Evaluation

This project presents a predictive approach to assessing oral cancer likelihood based on microbiome profiles. The model analyzes microbial patterns and provides a probability-based prediction that supports non-invasive diagnostic decision-making.

To further enhance the predictive value—especially for anticipating cancer **before** clinical onset—future iterations could integrate **longitudinal data**, enabling time-aware modeling and early detection frameworks. Incorporating methods like **survival analysis**, **Cox regression**, or **deep learning-based time-to-event modeling** would support forecasting the potential onset or progression of oral cancer more precisely over time.

* Random Forest is trained with `class_weight='balanced'`
* Evaluation is done via:

  * Accuracy: **92.89%**
  * AUROC: **0.9714**
  * PR-AUC: **0.9588**

### Key Visual Outputs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/5cea0344-c120-402d-8074-2ae8590372e3" width="300" alt="confusion_matrix"/>
  <img src="https://github.com/user-attachments/assets/186b44e7-361d-44e8-99f5-457a57e2e55e" width="300" alt="roc_curve"/>
  <img src="https://github.com/user-attachments/assets/53f14d6f-0f22-440c-a5d4-4e4540b5df5b" width="387" alt="shap_summary_plot"/>
</p>

---

### Run the full pipeline:

```bash
python main.py
```

Or step through it via notebook:

```bash
notebooks/_01_OralCancer_Modeling.ipynb
```

---

## 📂 Project Structure

```
OralCancerPrediction/
├── data/
│   ├── raw/                  # Place downloaded TCMA files here
│   └── processed/            # Outputs from preprocessing
│       ├── merged_with_labels.csv
│       └── selected_features.txt
├── notebooks/               # Jupyter notebooks
│   ├── _00_brief.ipynb
│   ├── _01_OralCancer_Modeling.ipynb
│   ├── _02_SHAP_Explainability.ipynb
│   └── _03_Deployment_Testing.ipynb
├── models/
│   └── rf_model.pkl          # Trained Random Forest model
├── outputs/                 # Evaluation results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary_plot.png
│   ├── metrics_summary.json
│   └── metrics_summary.txt
├── src/                     # Source code
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── main.py                  # Main script to run pipeline
├── requirements.txt
└── README.md
```

---

## 🧠 SHAP Explainability

* Global feature importance is visualized using `summary_plot`
* Additional force plots explain individual predictions

See: `notebooks/_02_SHAP_Explainability.ipynb`

---

## 👤 Contributor

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
      <a href="https://github.com/Mazenmarwan023" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
        <br />
        <sub><b>Mazen Marwan</b></sub>
      </a>
    </td>    
    <td align="center">
      <a href="https://github.com/madonna-mosaad" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="150px;" alt="Madonna Mosaad"/>
        <br />
        <sub><b>Madonna Mosaad</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>

---

## 📌 Note

This repository is for academic and demonstration purposes only. Ensure you comply with the TCMA terms of use when downloading data.
