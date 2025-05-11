# src/_04_utils.py
import joblib
import json


def save_model(model, path: str):
    """
    Serialize and save model to disk.
    """
    joblib.dump(model, path)


def save_json(data, path: str):
    """
    Save dictionary data as JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
