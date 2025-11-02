# scripts/predict_with_probs.py
# import os
# import joblib
# import numpy as np
# from scripts.preprocessing import clean_text  # <-- adjust if your file is at root: `from preprocessing import clean_text`

# MODEL_DIR = "models"
# VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
# LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

# # Load once at import time
# vectorizer = joblib.load(VECTORIZER_PATH)
# lr = joblib.load(LR_PATH)
# CLASSES = lr.classes_.tolist()  # order matches predict_proba columns

# def sanitize(name: str) -> str:
#     """Make safe column suffixes."""
#     return name.replace(" ", "_").replace("-", "_")

# def predict_with_probs(text: str):
#     """
#     Returns:
#       pred_label: str
#       probs_dict: {class_name: probability_float_0_to_1}
#     """
#     clean = clean_text(text)
#     # print(clean)
#     X = vectorizer.transform([clean])
#     proba = lr.predict_proba(X)[0]   # np.array aligned with lr.classes_
#     pred_idx = int(np.argmax(proba))
#     pred_label = CLASSES[pred_idx]
#     probs_dict = {c: float(p) for c, p in zip(CLASSES, proba)}
#     return pred_label, probs_dict, CLASSES



# final scripts/predict_with_probs_miniLM.py
import os
import joblib
import json
import numpy as np
import requests

# ===== Config =====
HF_REPO = "Ranajoy66/emotion-analysis-models"  # 🔹 your Hugging Face repo name
MODEL_DIR = "models/minilm_emotion"
FILES = ["embedder.pkl", "logistic_regression.pkl", "label2id.json"]

# ===== Ensure model folder exists =====
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_hf_if_needed():
    """Download model files from Hugging Face Hub if not already cached locally."""
    base_url = f"https://huggingface.co/{HF_REPO}/resolve/main/"
    
    for fname in FILES:
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⬇️ Downloading {fname} from Hugging Face...")
            url = base_url + fname
            r = requests.get(url)
            if r.status_code == 200:
                with open(fpath, "wb") as f:
                    f.write(r.content)
                print(f"✅ Saved: {fpath}")
            else:
                raise RuntimeError(f"❌ Failed to download {fname} (status {r.status_code})")

# ===== Load models =====
print("⏳ Checking and loading MiniLM embedder and Logistic Regression model...")
download_from_hf_if_needed()

EMBEDDER_PATH = os.path.join(MODEL_DIR, "embedder.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
LABEL2ID_PATH = os.path.join(MODEL_DIR, "label2id.json")

embedder = joblib.load(EMBEDDER_PATH)
lr_calibrated = joblib.load(LR_PATH)

with open(LABEL2ID_PATH, "r") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
CLASSES = sorted(label2id.keys(), key=lambda x: label2id[x])

# ===== Prediction function =====
def predict_with_probs(text: str):
    """
    Returns:
        pred_label: str
        probs_dict: {class_name: probability_float_0_to_1}
        classes_list: list of class names
    """
    emb = embedder.encode([text])
    proba = lr_calibrated.predict_proba(emb)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = id2label[pred_idx]

    probs_dict = {id2label[i]: float(p) for i, p in enumerate(proba)}

    return pred_label, probs_dict, CLASSES
