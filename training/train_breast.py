import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import load_dataset

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "breast_cancer.csv")

# Model save path
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "breast_model.pkl")

# Load dataset
df = load_dataset(DATASET_PATH)

# Target column
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Breast Cancer Model Accuracy: {accuracy*100:.2f}%")

# Create folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Save model
joblib.dump(model, MODEL_PATH)

print(f"Model saved successfully at: {MODEL_PATH}")