from sklearn.ensemble import RandomForestClassifier
from typing import Any
import joblib
import os

class OralCancerModel:
    """
    Model wrapper for Oral Cancer Prediction using Random Forest.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        class_weight: str = 'balanced',
        max_depth: int = None,
        seed: int = 42
    ):
        """
        Args:
            n_estimators: The number of trees in the forest.
            class_weight: Weights associated with classes.
            max_depth: The maximum depth of the tree.
            seed: Random seed.
        """
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.seed = seed
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            max_depth=self.max_depth,
            random_state=self.seed
        )

    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save the underlying model."""
        joblib.dump(self.model, path)
