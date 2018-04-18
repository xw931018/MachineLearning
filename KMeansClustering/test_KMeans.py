from KMeansClustering import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
import unittest
import numpy as np

class KMeansTestCase(unittest.TestCase):
    """Unit tests for KMeans++ algorithm"""

    def test_basic(self):
        X, y_true = make_blobs(n_samples = 500, n_features = 5, centers = 4, cluster_std = 0.6)
        km = KMeans.KMeansPlus(K = 4)
        self.assertEqual(km._K, 4)
        result = km.fit(data = X, iterations = 500, initial_method = "kmeans++")
        y_pred = result[1]
        y_true_transformed = np.array([dict(zip(y_true, y_pred))[i] for i in y_true])
        acc = accuracy_score(y_true_transformed, y_pred)
        self.assertGreaterEqual(acc, 0.99999)

if __name__ == "main":
    unittest.main()