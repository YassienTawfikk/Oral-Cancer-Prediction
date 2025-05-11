import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """
    Load merged feature-label CSV and split into X (features) and y (label).
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def split_data(X, y, test_size: float = 0.1, random_state: int = 42):
    """
    Stratified train-test split maintaining class ratio.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
