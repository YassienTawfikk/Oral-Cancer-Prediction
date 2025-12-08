import joblib
import json
import logging
from typing import Any

def save_model(model: Any, path: str) -> None:
    """
    Serialize and save model to disk.
    
    Args:
        model: The model object to save.
        path: Destination path.
    """
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model to {path}: {e}")
        raise

def save_json(data: dict, path: str) -> None:
    """
    Save dictionary data as JSON file.
    
    Args:
        data: Dictionary to save.
        path: Destination path.
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save JSON to {path}: {e}")
        raise
