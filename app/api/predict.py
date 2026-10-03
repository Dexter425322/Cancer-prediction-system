from fastapi import APIRouter
from app.schemas.prediction_schema import PredictionRequest
from app.services.prediction_service import (
    predict_breast,
    predict_cervical,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/breast")
def breast_prediction(data: PredictionRequest):
    return predict_breast(data.features)


@router.post("/cervical")
def cervical_prediction(data: PredictionRequest):
    return predict_cervical(data.features)