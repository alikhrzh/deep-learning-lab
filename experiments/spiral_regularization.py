"""Compare an unregularized MLP with L2 + dropout on noisy spiral data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_learning_lab import MLP, make_spiral


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def main() -> None:
    x, y = make_spiral(samples_per_class=160, noise=0.24, seed=11)
    split = 240
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    configurations = {
        "baseline": MLP(2, hidden_dim=64, seed=11),
        "l2 + dropout": MLP(2, hidden_dim=64, l2=0.03, dropout=0.05, seed=11),
    }

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=18)
    axes[0].set_title("Training data")

    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    for axis, (label, model) in zip(axes[1:], configurations.items()):
        history = model.fit(x_train, y_train, epochs=5000, learning_rate=0.1)
        surface = model.predict_proba(grid).reshape(grid_x.shape)
        axis.contourf(grid_x, grid_y, surface, levels=20, cmap="coolwarm", alpha=0.75)
        axis.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm", edgecolor="white", s=24)
        axis.set_title(f"{label}\ntest accuracy: {accuracy(model, x_test, y_test):.3f}")
        print(f"{label}: final loss={history[-1]:.4f}, test accuracy={accuracy(model, x_test, y_test):.3f}")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    output = Path(__file__).resolve().parents[1] / "assets" / "spiral_regularization.png"
    output.parent.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
