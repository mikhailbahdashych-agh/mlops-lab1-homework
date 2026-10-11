from pathlib import Path
from typing import Annotated

import joblib
from cleantext import clean
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).parent / "model"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")

encoder = SentenceTransformer(str(MODEL_DIR / "sentence_transformer.model"))
classifier = joblib.load(MODEL_DIR / "classifier.joblib")


class PredictionRequest(BaseModel):
    text: NonEmptyText


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    embedding = encoder.encode([clean(request.text)])
    label = int(classifier.predict(embedding)[0])
    return PredictionResponse(prediction=LABELS[label])
