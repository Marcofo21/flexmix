"""Main mixture model interface."""

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import optimagic as om


class MixtureModel:
    """Finite mixture model with flexible component likelihoods.

    Args:
        component_loglik: Function that computes log-likelihood for one component.
            Signature: (X, params) -> array of log-likelihoods
        n_components: Number of mixture components.
        method: Estimation method. One of ['em', 'direct'].
        optimizer: Optimagic algorithm name or callable.
        optimizer_kwargs: Additional arguments for optimizer.

    """

    def __init__(
        self,
        component_loglik: Callable[[NDArray, dict], NDArray],
        n_components: int = 2,
        method: str = "em",
        optimizer: str = "scipy_lbfgsb",
        optimizer_kwargs: dict | None = None,
    ):
        _fail_if_invalid_method(method)
        _fail_if_invalid_n_components(n_components)

        self.component_loglik = component_loglik
        self.n_components = n_components
        self.method = method
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.params_ = None
        self.weights_ = None

    def fit(
        self,
        X: NDArray,
        initial_model_params: list[dict],
        initial_group_probs: NDArray | None = None,
    ) -> "MixtureModel":
        """Estimate mixture model parameters.

        Args:
            X: Data array of shape (n_samples, n_features).
            initial_model_params: List of parameter dicts, one per component.
            initial_group_probs: Initial mixture weights. If None, uses uniform weights.

        Returns:
            self

        """
        _fail_if_invalid_model_params(initial_model_params, self.n_components)

        if initial_group_probs is None:
            initial_group_probs = _initialize_uniform_weights(self.n_components)
        else:
            _fail_if_invalid_group_probs(initial_group_probs, self.n_components)

        if self.method == "em":
            params, weights = estimate_em(
                X,
                initial_model_params,
                initial_group_probs,
                self.component_loglik,
                self.optimizer,
                self.optimizer_kwargs,
            )
        elif self.method == "direct":
            pass

        self.params_ = params
        self.weights_ = weights
        return self

    def predict_membership(self, X: NDArray) -> NDArray:
        """Predict component membership.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            Array of component assignments.

        """
        # TODO: implement
        pass

    def predict_probabilities(self, X):
        """Predict component membership probabilities.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_components) with membership probabilities.

        """
        # TODO: implement
        pass

    def score(self, X):
        """Compute log-likelihood of data.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            Log-likelihood value.

        """
        # TODO: implement
        pass


def _fail_if_invalid_method(method):
    valid = ["em", "direct"]
    if method not in valid:
        msg = f"method must be one of {valid}, got '{method}'"
        raise ValueError(msg)


def _fail_if_invalid_n_components(n_components):
    if not isinstance(n_components, int):
        msg = f"n_components must be int, got {type(n_components)}"
        raise TypeError(msg)
    if n_components < 1:
        msg = f"n_components must be >= 1, got {n_components}"
        raise ValueError(msg)
