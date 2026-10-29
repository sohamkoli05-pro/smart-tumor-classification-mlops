from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/best_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Smart Tumor Classification API",
    description="Educational machine learning classification API. Not for medical diagnosis.",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "Smart Tumor Classification API is running",
        "features_required": int(model.n_features_in_)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionInput):
    expected = int(model.n_features_in_)

    if len(request.features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, received {len(request.features)}"
        )

    feature_names = list(model.feature_names_in_)
    input_data = pd.DataFrame([request.features], columns=feature_names)
    prediction = int(model.predict(input_data)[0])

    return {
        "prediction": prediction,
        "class_name": "malignant" if prediction == 0 else "benign",
        "notice": "Educational demonstration only. Not for medical diagnosis."
    }
