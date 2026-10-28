"""FastAPI serving layer for the deterministic spiral classifier."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import make_spiral
from .mlp import MLP


TRAINING_SEED = 11
TRAINING_SAMPLES = 240


def train_model() -> MLP:
    features, labels = make_spiral(samples_per_class=160, noise=0.24, seed=TRAINING_SEED)
    model = MLP(2, hidden_dim=64, seed=TRAINING_SEED)
    model.fit(features[:TRAINING_SAMPLES], labels[:TRAINING_SAMPLES], epochs=5000, learning_rate=0.1)
    return model


model = train_model()
app = FastAPI(title="Deep Learning Lab API", version="0.1.0")


class Point(BaseModel):
    x1: float = Field(description="First normalized spiral coordinate")
    x2: float = Field(description="Second normalized spiral coordinate")


class Prediction(BaseModel):
    probability: float
    label: int


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "model": "numpy-mlp",
        "training_seed": TRAINING_SEED,
        "training_samples": TRAINING_SAMPLES,
    }


@app.post("/predict", response_model=Prediction)
def predict(point: Point) -> Prediction:
    probability = float(model.predict_proba([[point.x1, point.x2]])[0])
    return Prediction(probability=probability, label=int(probability >= 0.5))
