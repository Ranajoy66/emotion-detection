import pandas as pd
import os
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scripts.preprocessing import clean_text

# ===== Paths =====
DATA_PATH = "data/finaldata_filtered.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ===== Load dataset =====
df = pd.read_csv(DATA_PATH, encoding="latin1")

# Clean text
df["clean_text"] = df["Text"].apply(clean_text)

texts = df["clean_text"]
labels = df["Label"]  # your dataset already has this column

# ===== TF-IDF =====
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# ===== Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# ===== Train Logistic Regression =====
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# ===== Evaluation =====
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"✅ Training Complete | Accuracy: {acc*100:.2f}%")

# ===== Save models =====
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(lr, LR_PATH)

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump({"accuracy": acc, "report": report, "classes": lr.classes_.tolist()}, f, indent=4)

print(f"📊 Metrics saved at {METRICS_PATH}")
