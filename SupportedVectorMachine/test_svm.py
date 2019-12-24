import unittest
from SupportedVectorMachine import svm
from sklearn.datasets.samples_generator import  make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVMTestCase(unittest.TestCase):
    """Tests for SVM classifier"""

    def generate_blobs_sample(self):
        X, y = make_blobs(n_samples = 1000, n_features = 5, centers = 2, cluster_std = 1.8)
        y = (y - 0.5)*2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        return X_train, X_test, y_train, y_test

    def generate_moons_sample(self):
        X, y = make_moons(n_samples = 500, noise = 0.08)
        y = (y - 0.5) * 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def test_basic_blobs(self):
        X_train, X_test, y_train, y_test = self.generate_blobs_sample()
        svm = svm.SVM_Linear(X_train, y_train)
        svm.fit()
        test_pred = svm.classify(X_test)
        acc_train = accuracy_score(y_train, svm.classify(X_train))
        acc_test = accuracy_score(y_test, test_pred)
        self.assertGreater(acc_train, 0.999)
        self.assertGreater(acc_test, 0.999)
        for idx, x in enumerate(test_pred):
            self.assertEqual(x, y_test[idx])

    def test_basic_moons(self):
        X_train, X_test, y_train, y_test = self.generate_moons_sample()
        svm = svm.SVM_Linear(X_train, y_train, kernel = svm.RBF)
        svm.fit()
        test_pred = svm.classify(X_test)
        acc_train = accuracy_score(y_train, svm.classify(X_train))
        acc_test = accuracy_score(y_test, test_pred)
        self.assertGreater(acc_train, 0.9)
        self.assertGreater(acc_test, 0.9)

if __name__ == "main":
    unittest.main()

