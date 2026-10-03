from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI(
    title="Cancer Prediction API",
    description="API for Breast and Cervical Cancer Prediction",
    version="1.0.0"
)

app.include_router(predict_router)


@app.get("/")
def home():
    return {
        "message": "Cancer Prediction API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "Server is running successfully"
    }