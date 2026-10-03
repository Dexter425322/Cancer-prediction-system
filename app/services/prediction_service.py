from app.models.model_loader import breast_model, cervical_model


def predict_breast(features):
    prediction = breast_model.predict([features])[0]
    probability = breast_model.predict_proba([features]).max()

    return {
        "prediction": str(prediction),
        "confidence": float(probability)
    }


def predict_cervical(features):
    prediction = cervical_model.predict([features])[0]
    probability = cervical_model.predict_proba([features]).max()

    return {
        "prediction": str(prediction),
        "confidence": float(probability)
    }