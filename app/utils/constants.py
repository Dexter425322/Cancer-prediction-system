import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BREAST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "breast_model.pkl"
)

CERVICAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cervical_model.pkl"
)