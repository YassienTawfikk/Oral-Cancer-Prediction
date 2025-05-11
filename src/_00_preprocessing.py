"""
Full preprocessing pipeline for TCMA microbiome data

Steps
-----
1. merge_raw_data():   Merge abundance + metadata tables -> clean CSVs
2. DataPreprocessor:   Pre-processing, scaling, feature selection
3. main():             Run everything, save outputs
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# ──────────────────────────────  CONFIG  ────────────────────────────── #
RAW_INPUT_DIR = "../data/raw/TCMA/tb09j6496"
CLEAN_DATA_DIR = "../data/raw/TCMA/clean data"
FINAL_OUTPUT_DIR = "../data/processed"
DROP_FEATURE = "1678.0"
N_FEATURES = 17  # number of features to keep with SFS
# ────────────────────────────────────────────────────────────────────── #

# -----------------------------------------------------------------------------
# 0. House-keeping
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------------------------------------------------------
# 1. Merge raw abundance + metadata files
# -----------------------------------------------------------------------------
def merge_raw_data():
    """Create cleaned, per-cohort CSVs in CLEAN_DATA_DIR."""
    configs = [
        ("WGS", "blood", "sample", "clr"),
        ("WGS", "blood", "case", "clr"),
        ("WGS", "solid", "sample", "clr"),
        ("WGS", "solid", "case", "clr"),
        ("WXS", "blood", "sample", "clr"),
        ("WXS", "blood", "case", "clr"),
        ("WXS", "solid", "sample", "clr"),
        ("WXS", "solid", "case", "clr"),
    ]

    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

    for seq, tissue, level, ext in configs:
        try:
            abundance_f = f"{RAW_INPUT_DIR}/bacteria.{seq}.{tissue}.{level}.{ext}.txt"
            metadata_f = f"{RAW_INPUT_DIR}/metadata.{seq}.{tissue}.{level}.txt"
            out_csv = f"{CLEAN_DATA_DIR}/merged_{seq}_{tissue}_{level}_{ext}.csv"

            # Abundance: samples in rows after transpose
            X = pd.read_csv(abundance_f, sep="\t", index_col=0).T

            # Metadata: rename first col to SampleID for join
            meta = pd.read_csv(metadata_f, sep="\t")
            meta.rename(columns={meta.columns[0]: "SampleID"}, inplace=True)

            df = X.merge(meta, left_index=True, right_on="SampleID")

            # put metadata columns first
            meta_cols = meta.columns.tolist()
            feature_cols = [c for c in df.columns if c not in meta_cols]
            df = df[meta_cols + feature_cols]

            df.to_csv(out_csv, index=False)
            logging.info(f"✔  Merged & saved → {out_csv}")
        except Exception as e:
            logging.error(f"✘  Error merging {seq}-{tissue}-{level}-{ext}: {e}")


# -----------------------------------------------------------------------------
# 2. Pre-processing + feature selection
# -----------------------------------------------------------------------------
class DataPreprocessor:
    def __init__(self, data_dir=CLEAN_DATA_DIR):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.selected_features = None

    # 2-a. merge per-cohort CSVs into one dataframe
    def load_and_merge_files(self, csv_files, out_name="merged_all_data.csv"):
        merged = pd.DataFrame()
        for f in csv_files:
            try:
                path = os.path.join(self.data_dir, f)
                df = pd.read_csv(path, low_memory=False)
                df["data_source"] = f
                merged = pd.concat([merged, df], ignore_index=True)
                logging.info(f"→ merged {f}")
            except Exception as err:
                logging.error(f"Could not load {f}: {err}")

        if merged.empty:
            logging.error("All merges failed – no data.")
            return None

        merged.to_csv(os.path.join(self.data_dir, out_name), index=False)
        return merged

    # 2-b. clean, impute, encode, scale
    def preprocess(self, df):
        if df is None:
            return None, None

        # numeric medians
        for col in df.select_dtypes(np.number):
            df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)

        # categorical modes
        cat_cols = df.select_dtypes("object").columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            if col not in ["SampleID", "data_source"]:
                df[col] = self.encoder.fit_transform(df[col].astype(str))

        # choose target
        possible_targets = [
            "person_neoplasm_cancer_status", "vital_status",
            "tumor_tissue_site", "icd_10", "icd_o_3_site"
        ]
        target_col = next((c for c in possible_targets if c in df.columns), None)
        if target_col is None:
            logging.error("No target column found.")
            return None, None

        # split X / y
        drop_cols = [
            target_col, "SampleID", "data_source", "acronym",
            "days_to_birth", "days_to_death", "days_to_last_followup",
            "days_to_initial_pathologic_diagnosis",
            "tissue_retrospective_collection_indicator",
            "tissue_prospective_collection_indicator",
            "project_code", "patient_id", "year_of_initial_pathologic_diagnosis",
            "tissue_source_site", "form_completion_date", "system_version",
            "tss_site", "tss_study", "tss_bcr",
            "TSS_tss_site", "TSS_tss_study", "TSS_tss_bcr",
            "city_of_procurement", "country_of_procurement",
            "state_province_of_procurement", "state_province_country_of_procurement"
        ]
        X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        y = df[target_col]

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        return X_scaled, y

    # 2-c. SFS feature selection
    def select(self, X, y, k=N_FEATURES):
        try:
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            model = LogisticRegression(max_iter=5000)
            sfs = SFS(model, k_features=k, forward=True, floating=False,
                      scoring="accuracy", cv=5, verbose=0)
            sfs.fit(X_train.values, y_train)
            self.selected_features = [X_train.columns[i] for i in sfs.k_feature_idx_]
            logging.info(f"Selected features ({k}): {self.selected_features}")
            return X[self.selected_features]
        except Exception as err:
            logging.error(f"SFS failed: {err}")
            self.selected_features = X.columns.tolist()
            return X


# -----------------------------------------------------------------------------
# 3. Main pipeline
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. Main pipeline
# -----------------------------------------------------------------------------
def main():
    # Ensure processed-data folder exists (for feature-selected CSVs later)
    os.makedirs("../data/processed", exist_ok=True)

    # ── 1. Merge raw abundance + metadata ───────────────────────────────────
    merge_raw_data()

    # ── 2. Pre-process & select top-k features ──────────────────────────────
    pre = DataPreprocessor()
    merged_csvs = [
        "merged_WXS_solid_case_clr.csv",
        "merged_WXS_blood_case_clr.csv",
        "merged_WGS_solid_case_clr.csv",
        "merged_WGS_blood_case_clr.csv",
    ]
    big_df = pre.load_and_merge_files(merged_csvs)
    X, y = pre.preprocess(big_df)
    y.name = "label"
    if X is None or y is None:
        return

    # Feature selection (k = N_FEATURES from config)
    X_sel = pre.select(X, y)

    # ── 3. Final clean-up — drop feature “1678.0” once & for all ─────────────
    X_sel.columns = X_sel.columns.map(lambda c: str(c).strip())
    DROP_FEATURE_ACTUAL = "1678.0"
    X_sel = X_sel.drop(columns=[DROP_FEATURE_ACTUAL], errors="ignore")
    pre.selected_features = [
        f for f in pre.selected_features if str(f).strip() != DROP_FEATURE_ACTUAL
    ]
    assert DROP_FEATURE_ACTUAL not in X_sel.columns, "Feature 1678.0 still present!"

    # ── 4. Save outputs ─────────────────────────────────────────────────────
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    # Combine X and y for final output
    final_df = pd.concat([X_sel, y], axis=1)
    merged_out = os.path.join(FINAL_OUTPUT_DIR, "merged_with_labels.csv")
    final_df.to_csv(merged_out, index=False)
    logging.info(f"Saved → {merged_out}")

    # Also print the full cleaned table
    print("\n✅ Final cleaned dataset (with '1678.0' removed):")
    print(final_df.head())

    # Save selected feature list
    feat_list_file = os.path.join(FINAL_OUTPUT_DIR, "selected_features.txt")
    with open(feat_list_file, "w") as fh:
        fh.write("\n".join(pre.selected_features))
    logging.info(f"Saved feature list → {feat_list_file}")


if __name__ == "__main__":
    main()
