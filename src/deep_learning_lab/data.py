"""Synthetic datasets used by the experiments."""

from __future__ import annotations

import numpy as np


def make_spiral(samples_per_class: int = 120, noise: float = 0.2, seed: int = 7):
    """Return a two-class spiral dataset with shape ``(2*n, 2)``."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    features = []
    labels = []
    radius = np.linspace(0.05, 1.0, samples_per_class)

    for class_id in (0, 1):
        angle = np.linspace(class_id * np.pi, (class_id + 2.5) * np.pi, samples_per_class)
        angle += rng.normal(0.0, noise, samples_per_class)
        features.append(np.column_stack((radius * np.sin(angle), radius * np.cos(angle))))
        labels.append(np.full(samples_per_class, class_id, dtype=float))

    x = np.vstack(features)
    y = np.concatenate(labels)
    order = rng.permutation(len(y))
    return x[order], y[order]
