import numpy as np

from deep_learning_lab import MLP, gradient_check, make_spiral


def test_gradient_check_is_precise():
    x = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8], [0.5, 0.6]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = MLP(2, hidden_dim=3, l2=0.03, seed=3)
    assert gradient_check(model, x, y) < 1e-5


def test_training_reduces_loss_and_learns_spiral_signal():
    x, y = make_spiral(samples_per_class=80, noise=0.15, seed=5)
    model = MLP(2, hidden_dim=64, l2=0.01, seed=5)
    initial_loss, _ = model.loss_and_gradients(x, y)
    history = model.fit(x, y, epochs=4000, learning_rate=0.1)
    assert history[-1] < initial_loss * 0.5
    assert np.mean(model.predict(x) == y) > 0.87


def test_spiral_generation_is_reproducible():
    first = make_spiral(seed=13)
    second = make_spiral(seed=13)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
