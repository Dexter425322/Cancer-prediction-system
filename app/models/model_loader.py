import joblib
from app.utils.constants import BREAST_MODEL_PATH, CERVICAL_MODEL_PATH

breast_model = joblib.load(BREAST_MODEL_PATH)
cervical_model = joblib.load(CERVICAL_MODEL_PATH)