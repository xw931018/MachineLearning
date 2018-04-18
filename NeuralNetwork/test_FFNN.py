import unittest
import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork_Feed_Forward as FFNN
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.datasets.samples_generator import  make_blobs
from sklearn.model_selection import train_test_split

class FFNNTestCase(unittest.TestCase):
    """tests for feed-forward neural network"""

    def basic_network_properties(self, network, expected_layers_shape):
        self.assertEquals(network._shape, expected_layers_shape)

    def matrix_scores_assertions(self, network, test, obs, prob = 0.9):
        preds = network.predict(test)
        acc, precs, recall = accuracy_score(obs, preds), \
                             precision_score(obs, preds), \
                             recall_score(obs, preds)
        self.assertGreaterEqual(acc, prob)
        self.assertGreaterEqual(precs, prob)
        self.assertGreaterEqual(recall, prob)

    def test_logistic_regression(self):
        """Our FFNN should correctly identifies logistic classification problem when using a sigmoid activation"""
        w = 0.5;
        b = 1
        x = w * np.array(range(20)) + b
        u = (FFNN.logistic(x) > FFNN.logistic(6 * w + b)) * 1
        test = pd.DataFrame(x.reshape(-1, 1), columns=['x'])
        test['y'] = u
        net0 = FFNN.Network(data=test.drop('y', axis=1), labels=u, n_neurons=[])
        net0.fit(step=0.5, iterations=10000)
        self.basic_network_properties(net0, [1, 2])
        self.matrix_scores_assertions(net0, test.drop('y', axis = 1), u)

    def test_blobs(self):
        X, y_true = make_blobs(n_samples=300, centers=2)
        datasets = pd.DataFrame(X, columns=["X1", "X2"])
        datasets['y'] = y_true
        X_train, X_test, y_train, y_test = train_test_split(datasets.drop('y', axis=1),                                                 y_true, train_size=0.8)
        X_train = X_train.reset_index().drop("index", axis=1)
        X_test = X_test.reset_index().drop("index", axis=1)
        net1 = FFNN.Network(data=X_train, labels=y_train, n_neurons=[10, 10])
        net1.fit(step=0.1, iterations=2000)
        self.basic_network_properties(net1, [2, 10, 10, 2])
        self.matrix_scores_assertions(net1, X_test, y_test, 0.95)

if __name__ == "main":
    unittest.main()