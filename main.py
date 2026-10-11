from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")


class PredictionRequest(BaseModel):
    text: NonEmptyText


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(prediction="positive")
