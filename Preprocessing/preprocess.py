import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import os
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    def __init__(self, data_dir='../Data/TCMA/Clean Data'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None

    def load_and_merge_files(self, file_list, output_filename='merged_all_data.csv'):
        """Load and merge multiple CSV files into one"""
        merged_data = pd.DataFrame()

        for file in file_list:
            try:
                file_path = os.path.join(self.data_dir, file)
                data = pd.read_csv(file_path, low_memory=False)

                # Add a column to indicate the source file
                data['data_source'] = file

                # Concatenate with the merged data
                merged_data = pd.concat([merged_data, data], axis=0, ignore_index=True)
                logging.info(f"Successfully merged {file}")

            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")

        # Save the merged file
        if not merged_data.empty:
            output_path = os.path.join(self.data_dir, output_filename)
            merged_data.to_csv(output_path, index=False)
            logging.info(f"Saved merged data to {output_path}")
            return merged_data
        else:
            logging.error("No data was merged - all files failed to load")
            return None

    def load_data(self, file_name):
        """Load data from CSV file"""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            # Set low_memory=False to handle mixed types
            data = pd.read_csv(file_path, low_memory=False)
            logging.info(f"Successfully loaded {file_name}")
            return data
        except Exception as e:
            logging.error(f"Error loading {file_name}: {str(e)}")
            return None

    def preprocess_data(self, data):
        """Basic preprocessing steps"""
        if data is None:
            return None, None

        # Log initial shape
        logging.info(f"Initial data shape: {data.shape}")

        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = data[col].median()
            if pd.isna(median_val):
                median_val = 0  # If all values are NaN, use 0
            data[col] = data[col].fillna(median_val)

        # For categorical columns, fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = data[col].mode()
            if len(mode_val) > 0:
                data[col] = data[col].fillna(mode_val[0])
            else:
                data[col] = data[col].fillna('Unknown')

        # Log shape after handling missing values
        logging.info(f"Data shape after handling missing values: {data.shape}")

        # Handle categorical variables
        for col in categorical_cols:
            if col not in ['SampleID', 'data_source']:  # Don't encode the sample ID or data source
                data[col] = self.label_encoder.fit_transform(data[col].astype(str))

        # Try different target columns in order of preference
        target_columns = ['person_neoplasm_cancer_status', 'vital_status', 'tumor_tissue_site', 'icd_10',
                          'icd_o_3_site']
        target_col = None

        for col in target_columns:
            if col in data.columns:
                target_col = col
                logging.info(f"Using {col} as target variable")
                break

        if target_col is None:
            logging.error("No suitable target column found in data")
            return None, None

        # Separate features and target
        X = data.drop([target_col, 'SampleID', 'data_source'], axis=1)  # Remove target, sample ID, and data source
        y = data[target_col]

        # Log shapes before scaling
        logging.info(f"Features shape before scaling: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled, y

    def select_features(self, X, y, n_features_to_select=20):
        """Perform feature selection using Sequential Feature Selection"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Initialize the base model
            base_model = LogisticRegression(max_iter=5000)

            # Initialize the Sequential Feature Selector
            sfs = SFS(
                base_model,
                k_features=n_features_to_select,
                forward=True,
                floating=False,
                scoring='accuracy',
                verbose=2,
                cv=5
            )

            # Fit the selector
            sfs = sfs.fit(X_train.values, y_train)

            # Get selected feature names
            self.selected_features = [X_train.columns[idx] for idx in sfs.k_feature_idx_]
            logging.info(f"Selected features: {self.selected_features}")

            return X[self.selected_features]

        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            return X

    def process_all_files(self):
        """Process all data files by first merging them"""
        processed_data = {}

        # List of files to process
        files = [
            'merged_WXS_solid_case_clr.csv',
            'merged_WXS_blood_case_clr.csv',
            'merged_WGS_solid_case_clr.csv',
            'merged_WGS_blood_case_clr.csv'
        ]

        # First merge all files
        merged_data = self.load_and_merge_files(files)
        if merged_data is None:
            return processed_data

        # Now process the merged file
        logging.info("Processing merged data")

        # Preprocess data
        X, y = self.preprocess_data(merged_data)
        if X is None or y is None:
            return processed_data

        # Select features
        X_selected = self.select_features(X, y)

        # Store processed data
        processed_data['merged_all_data'] = {
            'X': X_selected,
            'y': y,
            'selected_features': self.selected_features
        }

        return processed_data


import os
import logging
import pandas as pd


def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Process all files (now merged)
    processed_data = preprocessor.process_all_files()

    # Save processed data
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    for file_name, data in processed_data.items():
        # Create base file name without extension if present
        base_name = os.path.splitext(file_name)[0]

        # Save features (X) as CSV
        features_file = os.path.join(output_dir, f'processed_{base_name}.csv')
        data['X'].to_csv(features_file, index=False)

        # Save target (y) as CSV
        target_file = os.path.join(output_dir, f'target_{base_name}.csv')
        pd.DataFrame(data['y']).to_csv(target_file, index=False)

        # Save selected features list
        features_list_file = os.path.join(output_dir, f'selected_features_{base_name}.txt')
        with open(features_list_file, 'w') as f:
            f.write('\n'.join(data['selected_features']))

        logging.info(f"Saved processed data to {features_file}")
        logging.info(f"Saved target to {target_file}")
        logging.info(f"Saved selected features to {features_list_file}")


if __name__ == "__main__":
    main()
