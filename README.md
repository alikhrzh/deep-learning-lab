# Deep Learning Lab

An original NumPy implementation of a two-layer neural network, with a reproducible experiment, numerical gradient verification, a FastAPI serving layer, and endpoint tests. It does not publish course notebooks or copied assignment code.

![Decision boundaries for the spiral experiment](assets/spiral_regularization.png)

## Problem

High-level frameworks make neural networks productive, but they can hide the mechanics that determine whether a model is correct. This project keeps the model small enough to inspect every forward, backward, regularization, and serving step.

## Approach

- generate a deterministic two-class spiral dataset;
- implement a two-layer MLP with NumPy;
- use He initialization, ReLU, binary cross-entropy, L2, and inverted dropout;
- compare an unregularized baseline with an L2 + dropout configuration;
- verify analytical gradients with central finite differences;
- serve the baseline through a typed FastAPI `/predict` endpoint.

## Results

The experiment uses seed `11`, 320 generated samples, and a fixed 240/80 train/test split.

| Configuration | Test accuracy | Final training objective |
| --- | ---: | ---: |
| Baseline: 64 hidden units | 0.8750 | 0.3653 |
| L2 `0.03` + dropout `0.05` | 0.8375 | 0.4432 |

The regularized model performs worse on this small split. This is reported as an observed bias-variance tradeoff, not as evidence that regularization is generally harmful. The gradient test reaches a maximum relative error of `2.00347094604e-10` on a fixed four-example check.

These numbers come from `experiments/spiral_regularization.py` and `tests/test_mlp.py`. They are synthetic educational results, not a production benchmark.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
pytest -q
python experiments/spiral_regularization.py
```

Run the API locally:

```bash
uvicorn deep_learning_lab.api:app --reload
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"x1": 0.25, "x2": -0.4}'
```

Run the container:

```bash
docker build -t deep-learning-lab .
docker run --rm -p 8000:8000 deep-learning-lab
```

## Project structure

```text
assets/                    generated experiment visualization
docs/                      short data-to-production plan
experiments/               reproducible evaluation entry point
src/deep_learning_lab/     model, data generator, and FastAPI app
tests/                     gradient, learning, and endpoint tests
Dockerfile                 CPU-only API image
```

The repository follows a small `src` layout so reusable code stays separate
from one-off experiments:

- `src/deep_learning_lab/data.py` generates the deterministic spiral dataset.
- `src/deep_learning_lab/mlp.py` contains the NumPy model, training loop, and
  numerical gradient checker.
- `src/deep_learning_lab/api.py` trains the lightweight baseline at startup and
  exposes `/health` and `/predict`.
- `experiments/spiral_regularization.py` compares the baseline with L2 +
  dropout and regenerates the visualization in `assets/`.
- `tests/test_mlp.py` checks learning, reproducibility, and gradients;
  `tests/test_api.py` checks the HTTP contract and request validation.
- `pyproject.toml`, `Dockerfile`, and `.github/workflows/tests.yml` define the
  package, container, and continuous-integration entry points.

## Limitations

- The generated spiral dataset is useful for debugging nonlinear boundaries, not for benchmarking real-world models.
- Full-batch gradient descent is intentionally simple and is not optimized for large datasets.
- The API trains the deterministic lightweight model at process start; a larger model should load a versioned artifact instead.
- The published accuracy is from one fixed split and has no uncertainty interval.
- No public deployment is claimed yet: `[TODO: deployed API URL]`.

## Roadmap

- [x] Implement the NumPy model and gradient checker.
- [x] Add a reproducible regularization experiment and visualization.
- [x] Add FastAPI, Dockerfile, and endpoint tests.
- [ ] Deploy the container and verify the public health endpoint.
- [ ] Add repeated-seed evaluation with confidence intervals.

The implementation was created with AI-assisted development and then validated with deterministic tests and numerical gradient checking. The underlying neural-network concepts were studied through coursework, but graded course solutions are intentionally excluded.
