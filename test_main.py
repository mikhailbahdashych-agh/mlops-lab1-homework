import pytest
from fastapi.testclient import TestClient

from main import app, classifier, encoder

client = TestClient(app)


def test_models_are_loaded() -> None:
    assert classifier.n_features_in_ == encoder.get_embedding_dimension()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What a great MLOps lecture, I am very satisfied", "positive"),
        ("This is the worst service I have ever used, awful", "negative"),
        ("The lecture starts at 10am in room 204", "neutral"),
    ],
)
def test_predict(text: str, expected: str) -> None:
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json() == {"prediction": expected}


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": 123}, {}])
def test_predict_rejects_invalid_input(payload: dict[str, object]) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]
