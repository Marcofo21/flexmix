import numpy as np


def initialize_uniform_weights(n_components: int) -> NDArray:
    """Initialize uniform group probabilities.

    Args:
        n_components: Number of mixture components.

    Returns:
        Array of shape (n_components,) with uniform probabilities.

    """

    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

    return np.full(shape=(n_components,), fill_value=1.0 / n_components)
