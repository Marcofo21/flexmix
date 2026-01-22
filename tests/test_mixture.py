"""Tests for mixture models."""

import numpy as np

from finmix.mixture import GaussianMixture


class TestMixtureModelBase:
    """Test the base MixtureModel class."""

    def test_initialization(self):
        """Test basic initialization."""
        model = GaussianMixture(n_components=3, max_iter=50, tol=1e-3, random_state=42)
        assert model.n_components == 3
        assert model.max_iter == 50
        assert model.tol == 1e-3
        assert model.random_state == 42
        assert not model.converged_
        assert model.n_iter_ == 0

    def test_validate_data_numpy(self):
        """Test data validation with numpy arrays."""
        model = GaussianMixture()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_validated = model._validate_data(X)
        assert isinstance(X_validated, np.ndarray)
        assert X_validated.shape == (3, 2)

    def test_validate_data_1d(self):
        """Test data validation with 1D arrays."""
        model = GaussianMixture()
        X = np.array([1, 2, 3, 4, 5])
        X_validated = model._validate_data(X)
        assert X_validated.shape == (5, 1)


class TestGaussianMixture:
    """Test the GaussianMixture class."""

    def test_fit_simple_2d(self):
        """Test fitting on simple 2D data with two clear clusters."""
        # Create two well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2) + np.array([0, 0])
        cluster2 = np.random.randn(50, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        assert gmm.weights_ is not None
        assert gmm.means_ is not None
        assert gmm.covariances_ is not None
        assert gmm.weights_.shape == (2,)
        assert gmm.means_.shape == (2, 2)
        assert gmm.covariances_.shape == (2, 2, 2)
        assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_predict_simple(self):
        """Test prediction on simple data."""
        np.random.seed(42)
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)

        assert labels.shape == (60,)
        assert set(labels) <= {0, 1}

    def test_predict_proba(self):
        """Test probability prediction."""
        np.random.seed(42)
        X = np.random.randn(20, 2)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        probas = gmm.predict_proba(X)

        assert probas.shape == (20, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_score(self):
        """Test log-likelihood computation."""
        np.random.seed(42)
        X = np.random.randn(20, 2)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        score = gmm.score(X)

        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)

    def test_aic_bic(self):
        """Test AIC and BIC computation."""
        np.random.seed(42)
        X = np.random.randn(50, 2)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        aic = gmm.aic_
        bic = gmm.bic_

        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert not np.isnan(aic)
        assert not np.isnan(bic)

    def test_convergence(self):
        """Test convergence behavior."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, random_state=42)
        gmm.fit(X)

        assert gmm.n_iter_ > 0
        assert gmm.n_iter_ <= 100

    def test_random_initialization(self):
        """Test random initialization method."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        gmm = GaussianMixture(n_components=2, init_method="random", random_state=42)
        gmm.fit(X)

        assert gmm.means_ is not None
        assert gmm.covariances_ is not None

    def test_1d_data(self):
        """Test fitting on 1D data."""
        np.random.seed(42)
        X = np.concatenate([np.random.randn(20) - 2, np.random.randn(20) + 2])

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)

        assert labels.shape == (40,)
        assert gmm.means_.shape == (2, 1)

    def test_single_component(self):
        """Test fitting with a single component."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        gmm = GaussianMixture(n_components=1, random_state=42)
        gmm.fit(X)

        assert gmm.weights_.shape == (1,)
        assert gmm.means_.shape == (1, 2)
        assert np.isclose(gmm.weights_[0], 1.0)

    def test_regularization(self):
        """Test covariance regularization."""
        np.random.seed(42)
        X = np.random.randn(20, 2)

        gmm = GaussianMixture(n_components=2, reg_covar=1e-3, random_state=42)
        gmm.fit(X)

        # Check that covariances are not singular
        for cov in gmm.covariances_:
            det = np.linalg.det(cov)
            assert det > 0

    def test_different_n_components(self):
        """Test with different numbers of components."""
        np.random.seed(42)
        X = np.random.randn(50, 2)

        for n_comp in [1, 2, 3, 5]:
            gmm = GaussianMixture(n_components=n_comp, random_state=42)
            gmm.fit(X)
            assert gmm.weights_.shape == (n_comp,)
            assert gmm.means_.shape == (n_comp, 2)

    def test_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        gmm1 = GaussianMixture(n_components=2, random_state=42)
        gmm1.fit(X)
        labels1 = gmm1.predict(X)

        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(X)
        labels2 = gmm2.predict(X)

        # Results should be identical with the same random state
        assert np.array_equal(labels1, labels2)
        assert np.allclose(gmm1.means_, gmm2.means_)
        assert np.allclose(gmm1.weights_, gmm2.weights_)

    def test_fit_predict_chain(self):
        """Test chaining fit and predict."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        gmm = GaussianMixture(n_components=2, random_state=42)
        labels = gmm.fit(X).predict(X)

        assert labels.shape == (30,)

    def test_n_parameters(self):
        """Test parameter counting."""
        gmm = GaussianMixture(n_components=2, random_state=42)
        X = np.random.randn(20, 3)
        gmm.fit(X)

        # For 2 components with 3 features:
        # weights: 1 (n_components - 1)
        # means: 2 * 3 = 6
        # covariances: 2 * 3 * 4 / 2 = 12
        # Total: 1 + 6 + 12 = 19
        assert gmm._n_parameters() == 19
