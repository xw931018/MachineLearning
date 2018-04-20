import unittest
from SupportedVectorMachine import SVM
from sklearn.datasets.samples_generator import  make_blobs, make_moons
from sklearn.model_selection import train_test_split

class SVMTestCase(unittest.TestCase):
    """Tests for SVM classifier"""

    def generate_blobs_sample(self):
        X, y = make_blobs(n_samples = 500, n_features = 5, centers = 2)
        y = (y - 0.5)*2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        return X_train, X_test, y_train, y_test

    def generate_moon_sample(self):
        X, y = make_moons(n_samples = 500, noise = 0.15)
        y = (y - 0.5) * 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def test_basic_blobs(self):
        X_train, X_test, y_train, y_test = self.generate_blobs_sample()
        svm = SVM.SVM_Linear(X_train, y_train)
        svm.fit()

