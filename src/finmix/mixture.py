"""Main mixture model interface."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2


class MixtureModel:
    """Base class for finite mixture models.

    This class provides the foundation for mixture model implementations,
    including parameter estimation via the Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    n_components : int, default=2
        The number of mixture components.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for the log-likelihood.
    random_state : int or None, default=None
        Random seed for reproducibility.
    init_method : str, default='kmeans'
        Initialization method: 'kmeans' or 'random'.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        The mixing weights for each component.
    converged_ : bool
        Whether the EM algorithm converged.
    n_iter_ : int
        Number of iterations performed.
    log_likelihood_ : float
        Log-likelihood of the fitted model.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init_method: str = "kmeans",
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_method = init_method

        # Fitted parameters
        self.weights_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = -np.inf

    def fit(self, X):
        """Fit the mixture model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        self._initialize_parameters(X)
        self._run_em(X)
        return self

    def predict(self, X):
        """Predict the component labels for the data samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X)
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        """Predict posterior probabilities of each component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        probas : ndarray of shape (n_samples, n_components)
            Posterior probabilities.
        """
        X = self._validate_data(X)
        return self._e_step(X)

    def score(self, X):
        """Compute the log-likelihood of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data.
        """
        X = self._validate_data(X)
        return self._compute_log_likelihood(X)

    @property
    def aic_(self):
        """Akaike Information Criterion."""
        n_params = self._n_parameters()
        return -2 * self.log_likelihood_ + 2 * n_params

    @property
    def bic_(self):
        """Bayesian Information Criterion."""
        n_params = self._n_parameters()
        n_samples = self._n_samples
        return -2 * self.log_likelihood_ + n_params * np.log(n_samples)

    def _validate_data(self, X):
        """Convert and validate input data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _initialize_parameters(self, X):
        """Initialize model parameters (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _initialize_parameters")

    def _e_step(self, X):
        """E-step: compute responsibilities (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _e_step")

    def _m_step(self, X, responsibilities):
        """M-step: update parameters (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _m_step")

    def _compute_log_likelihood(self, X):
        """Compute log-likelihood (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _compute_log_likelihood")

    def _n_parameters(self):
        """Return the number of free parameters (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _n_parameters")

    def _run_em(self, X):
        """Run the EM algorithm."""
        self._n_samples = X.shape[0]
        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)

            # Check for convergence
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                self.log_likelihood_ = log_likelihood
                break

            prev_log_likelihood = log_likelihood
        else:
            self.n_iter_ = self.max_iter
            self.log_likelihood_ = prev_log_likelihood
            warnings.warn("EM algorithm did not converge.", UserWarning)


class GaussianMixture(MixtureModel):
    """Gaussian Mixture Model.

    Represents data as a mixture of Gaussian distributions. Uses the
    Expectation-Maximization (EM) algorithm for parameter estimation.

    Parameters
    ----------
    n_components : int, default=2
        The number of mixture components.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for the log-likelihood.
    random_state : int or None, default=None
        Random seed for reproducibility.
    init_method : str, default='kmeans'
        Initialization method: 'kmeans' or 'random'.
    reg_covar : float, default=1e-6
        Regularization added to the diagonal of covariance matrices
        for numerical stability.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        The mixing weights for each component.
    means_ : ndarray of shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : ndarray of shape (n_components, n_features, n_features)
        The covariance matrix of each mixture component.
    converged_ : bool
        Whether the EM algorithm converged.
    n_iter_ : int
        Number of iterations performed.
    log_likelihood_ : float
        Log-likelihood of the fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from finmix.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gmm = GaussianMixture(n_components=2, random_state=0)
    >>> gmm.fit(X)
    >>> gmm.predict(X)
    array([0, 0, 0, 1, 1, 1])
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init_method: str = "kmeans",
        reg_covar: float = 1e-6,
    ):
        super().__init__(n_components, max_iter, tol, random_state, init_method)
        self.reg_covar = reg_covar
        self.means_ = None
        self.covariances_ = None

    def _initialize_parameters(self, X):
        """Initialize parameters using k-means or random initialization."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Initialize means
        if self.init_method == "kmeans":
            try:
                self.means_, _ = kmeans2(X, self.n_components, seed=self.random_state)
            except Exception:
                # Fallback to random if k-means fails
                indices = rng.choice(n_samples, self.n_components, replace=False)
                self.means_ = X[indices]
        else:
            indices = rng.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices]

        # Initialize covariances as identity matrices
        self.covariances_ = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )

    def _e_step(self, X):
        """E-step: compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )

        # Normalize responsibilities
        resp_sum = responsibilities.sum(axis=1, keepdims=True)
        resp_sum[resp_sum == 0] = 1e-10  # Avoid division by zero
        responsibilities /= resp_sum

        return responsibilities

    def _m_step(self, X, responsibilities):
        """M-step: update parameters."""
        n_samples, n_features = X.shape
        resp_sum = responsibilities.sum(axis=0)

        # Update weights
        self.weights_ = resp_sum / n_samples

        # Update means
        self.means_ = np.dot(responsibilities.T, X) / resp_sum[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k, np.newaxis] * diff
            self.covariances_[k] = np.dot(weighted_diff.T, diff) / resp_sum[k]
            # Add regularization
            self.covariances_[k] += self.reg_covar * np.eye(n_features)

    def _compute_log_likelihood(self, X):
        """Compute the log-likelihood of the data."""
        n_samples = X.shape[0]
        log_likelihood = 0.0

        for i in range(n_samples):
            prob = 0.0
            for k in range(self.n_components):
                pdf_value = self._gaussian_pdf(
                    X[i : i + 1], self.means_[k], self.covariances_[k]
                )
                # Ensure we get a scalar
                if isinstance(pdf_value, np.ndarray):
                    pdf_value = (
                        pdf_value.item() if pdf_value.size == 1 else pdf_value[0]
                    )
                prob += self.weights_[k] * pdf_value
            log_likelihood += np.log(prob + 1e-10)

        return float(log_likelihood)

    def _gaussian_pdf(self, X, mean, covariance):
        """Compute Gaussian probability density function."""
        n_features = X.shape[1]
        diff = X - mean

        try:
            cov_det = np.linalg.det(covariance)
            cov_inv = np.linalg.inv(covariance)

            normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
            exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)

            return normalization * np.exp(exponent)
        except np.linalg.LinAlgError:
            # Handle singular covariance matrix
            return np.zeros(X.shape[0])

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        n_features = self.means_.shape[1]
        # weights (n_components - 1) + means + covariances
        n_mean_params = self.n_components * n_features
        n_cov_params = self.n_components * n_features * (n_features + 1) / 2
        return int(self.n_components - 1 + n_mean_params + n_cov_params)
