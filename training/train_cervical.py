import os
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "cervical_cancer.csv")

# Model save path
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "cervical_model.pkl")

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Replace '?' with NaN
df.replace("?", pd.NA, inplace=True)

# Convert every column to numeric if possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where target is missing
df = df.dropna(subset=["Biopsy"])

# Features and Target
X = df.drop(columns=["Biopsy"])
y = df["Biopsy"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

# Create folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Save model
joblib.dump(model, MODEL_PATH)

print(f"Model saved successfully at: {MODEL_PATH}")