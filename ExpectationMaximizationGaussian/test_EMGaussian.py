import unittest
from sklearn.datasets.samples_generator import make_blobs
from ExpectationMaximizationGaussian import EMGaussian

class EMGaussianTestCase(unittest.TestCase):
    """basic unit tests for model-based clustering with Gaussian distribution"""

    def test_basic_EM(self):
        X, y_true = make_blobs(n_samples = 1000, n_features = 10, centers = 6)
        em = EMGaussian.EM_Gaussian(K = 6)
        em.fit(X)