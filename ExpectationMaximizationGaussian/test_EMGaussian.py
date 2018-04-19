import unittest
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from ExpectationMaximizationGaussian import EMGaussian
from sklearn.metrics import accuracy_score

class EMGaussianTestCase(unittest.TestCase):
    """basic unit tests for model-based clustering with Gaussian distribution"""

    def test_basic_EM(self):
        X, y_true = make_blobs(n_samples=300, n_features=5, centers=4, cluster_std=1)
        em = EMGaussian.EM_Gaussian(K = 4)
        em.fit(X, iterations=500)
        y_predicted = em.predict(X)
        y_true_transformed = np.array([dict(zip(y_true, y_predicted))[i] for i in y_true])
        acc = accuracy_score(y_true_transformed, y_predicted)
        self.assertGreaterEqual(acc, 0.999)

if __name__ == "main":
    unittest.main()