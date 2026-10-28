"""A two-layer binary classifier implemented with NumPy only."""

from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class MLP:
    """Two-layer MLP with ReLU, optional dropout, and L2 regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        l2: float = 0.0,
        dropout: float = 0.0,
        seed: int = 7,
    ) -> None:
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("layer dimensions must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.l2 = l2
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.params = {
            "w1": self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1)),
            "b2": np.zeros(1),
        }

    def forward(self, x: np.ndarray, training: bool = False):
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(z1, 0.0)
        mask = None
        if training and self.dropout:
            keep_probability = 1.0 - self.dropout
            mask = (self.rng.random(hidden.shape) < keep_probability) / keep_probability
            hidden = hidden * mask
        logits = (hidden @ self.params["w2"] + self.params["b2"]).ravel()
        return _sigmoid(logits), (x, z1, hidden, mask)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray, training: bool = False):
        probabilities, cache = self.forward(x, training=training)
        x_cached, z1, hidden, mask = cache
        count = len(y)
        epsilon = 1e-12
        data_loss = -np.mean(
            y * np.log(probabilities + epsilon)
            + (1.0 - y) * np.log(1.0 - probabilities + epsilon)
        )
        penalty = 0.5 * self.l2 * (
            np.sum(self.params["w1"] ** 2) + np.sum(self.params["w2"] ** 2)
        ) / count

        dlogits = (probabilities - y)[:, None] / count
        gradients = {
            "w2": hidden.T @ dlogits + self.l2 * self.params["w2"] / count,
            "b2": np.sum(dlogits, axis=0),
        }
        dhidden = dlogits @ self.params["w2"].T
        if mask is not None:
            dhidden *= mask
        dz1 = dhidden * (z1 > 0.0)
        gradients["w1"] = x_cached.T @ dz1 + self.l2 * self.params["w1"] / count
        gradients["b1"] = np.sum(dz1, axis=0)
        return float(data_loss + penalty), gradients

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 3000, learning_rate: float = 0.08):
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        history = []
        for _ in range(epochs):
            loss, gradients = self.loss_and_gradients(x, y, training=True)
            for name in self.params:
                self.params[name] -= learning_rate * gradients[name]
            history.append(loss)
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def gradient_check(model: MLP, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-5) -> float:
    """Return the largest relative error between analytic and numeric gradients."""
    _, analytic = model.loss_and_gradients(x, y)
    maximum_error = 0.0

    for name, values in model.params.items():
        numeric = np.zeros_like(values)
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + epsilon
            plus, _ = model.loss_and_gradients(x, y)
            values[index] = original - epsilon
            minus, _ = model.loss_and_gradients(x, y)
            values[index] = original
            numeric[index] = (plus - minus) / (2.0 * epsilon)

        denominator = np.maximum(1e-8, np.abs(numeric) + np.abs(analytic[name]))
        maximum_error = max(maximum_error, float(np.max(np.abs(numeric - analytic[name]) / denominator)))

    return maximum_error
