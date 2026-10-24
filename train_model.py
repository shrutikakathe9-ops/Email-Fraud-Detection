import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "phishing_email.csv"
MODEL_PATH = BASE_DIR / "email_fraud_model.pkl"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Check required columns
required_columns = {"text_combined", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Required columns missing. Found: {df.columns.tolist()}")

# Prepare data
df = df[["text_combined", "label"]].dropna()
df["text_combined"] = df["text_combined"].astype(str).str.strip()
df = df[df["text_combined"] != ""]

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df = df[df["label"].isin([0, 1])]
df["label"] = df["label"].astype(int)

print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

if df["label"].nunique() != 2:
    raise ValueError("Dataset must contain both labels: 0 and 1.")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df["text_combined"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Build and train the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
print("Training complete!")