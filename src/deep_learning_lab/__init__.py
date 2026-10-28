"""Small, reproducible deep-learning experiments built from first principles."""

from .data import make_spiral
from .mlp import MLP, gradient_check

__all__ = ["MLP", "gradient_check", "make_spiral"]
