from dataclasses import dataclass
import pandas as pd
import os
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from .config import FINAL_OUTPUT_DIR, TEST_SIZE, RANDOM_STATE

class OralCancerDataModule:
    """
    DataModule for Oral Cancer Prediction.
    Handles data loading, splitting, and preparation.
    """
    def __init__(
        self, 
        data_path: str = os.path.join(FINAL_OUTPUT_DIR, "merged_with_labels.csv"),
        test_size: float = TEST_SIZE,
        seed: int = RANDOM_STATE
    ):
        """
        Args:
            data_path: Path to the processed CSV file.
            test_size: Fraction of data to use for testing.
            seed: Random seed for splitting.
        """
        self.data_path = data_path
        self.test_size = test_size
        self.seed = seed
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def setup(self):
        """Load and split the data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}. Please run preprocessing first.")
            
        df = pd.read_csv(self.data_path)
        if "label" not in df.columns:
            raise ValueError("Column 'label' missing from dataset.")
            
        X = df.drop(columns=["label"])
        y = df["label"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.seed
        )
        
    def train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return training data."""
        if self.X_train is None:
            self.setup()
        return self.X_train, self.y_train

    def test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return test data."""
        if self.X_test is None:
            self.setup()
        return self.X_test, self.y_test
