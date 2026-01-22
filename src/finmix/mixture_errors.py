import numpy as np


def fail_if_invalid_model_params(model_params, n_components):
    """Raise an error if model parameters are invalid.

    Args:
        model_params: List of model parameters for each component.
        n_components: Expected number of components.
    Raises:
        ValueError: If model parameters are invalid.
    """

    if not isinstance(model_params, list):
        raise ValueError(
            """Model parameters must be provided as a list of parameters dictionaries, 
            one per component."""
        )
    if len(model_params) != n_components:
        raise ValueError(
            f"Number of model parameter sets {len(model_params)} does not match "
            f"number of components {n_components}."
        )
    for i, params in enumerate(model_params):
        if not isinstance(params, dict):
            raise ValueError(
                f"Model parameters for component {i} must be a dictionary."
            )


def fail_if_invalid_group_probs(group_probs, n_components):
    """Raise an error if group probabilities are invalid.

    Args:
        group_probs: Array of group probabilities.
        n_components: Expected number of components.
    Raises:
        ValueError: If group probabilities are invalid.
    """

    group_probs = np.asarray(group_probs)
    if group_probs.ndim != 1:
        raise ValueError("Group probabilities must be a one-dimensional array.")
    if len(group_probs) != n_components:
        raise ValueError(
            f"Group probabilities length {len(group_probs)} does not match "
            f"number of components {n_components}."
        )
    if np.any(group_probs < 0):
        raise ValueError("Group probabilities must be non-negative.")
    if not np.isclose(np.sum(group_probs), 1.0):
        raise ValueError("Group probabilities must sum to 1.")
