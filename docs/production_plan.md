# Data-to-production plan

Deep Learning Lab is the shortest path to a complete deployment because its dataset is generated deterministically, training is lightweight, and the model has no external service or model-download dependency.

1. Generate the labelled spiral dataset from a fixed seed.
2. Keep a fixed train/test split and record the evaluation method.
3. Train the NumPy MLP and verify gradients numerically.
4. Report the baseline and regularized-model results without selecting the better test result after the fact.
5. Expose a typed `/predict` endpoint through FastAPI.
6. Add health and request-validation tests.
7. Package the service in a small CPU-only Docker image.
8. Run tests and the experiment in CI.
9. Deploy the container to a service such as Render or Fly.io.
10. Add the public URL only after checking `/health` and one known `/predict` request.
