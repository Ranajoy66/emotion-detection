# scripts/predict_with_probs.py
import os
import joblib
import numpy as np

from scripts.preprocessing import clean_text  # <-- adjust if your file is at root: `from preprocessing import clean_text`

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

# Load once at import time
vectorizer = joblib.load(VECTORIZER_PATH)
lr = joblib.load(LR_PATH)
CLASSES = lr.classes_.tolist()  # order matches predict_proba columns

def sanitize(name: str) -> str:
    """Make safe column suffixes."""
    return name.replace(" ", "_").replace("-", "_")

def predict_with_probs(text: str):
    """
    Returns:
      pred_label: str
      probs_dict: {class_name: probability_float_0_to_1}
    """
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    proba = lr.predict_proba(X)[0]   # np.array aligned with lr.classes_
    pred_idx = int(np.argmax(proba))
    pred_label = CLASSES[pred_idx]
    probs_dict = {c: float(p) for c, p in zip(CLASSES, proba)}
    return pred_label, probs_dict, CLASSES
