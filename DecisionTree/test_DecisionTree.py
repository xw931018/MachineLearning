import pandas as pd
import unittest
import numpy as np
import random
import DecisionTree
from sklearn.metrics import precision_score, accuracy_score, recall_score

mushrooms = pd.read_csv("./mushrooms.csv")

class DecisionTreeTestCase(unittest.TestCase):

    """Test for basic output of the decision tree"""

    def generate_basic_test_data(self):
        features = pd.DataFrame([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 1, 1],
                             [1, 0, 0],
                             [1, 0, 1],
                             [1, 1, 0],
                             [1, 1, 1],
                             ], columns = ["X1", "X2", "X3"])
        target = random.shuffle(np.array([1]*4 + [0]*4))
        features['Y'] = target
        return features

    def test_basic_result(self):
        data = self.generate_basic_test_data(); X = data.drop('Y', axis = 1); y = data['Y']
        tree = DecisionTree.Tree(data = X,
                                 labels = y,
                                 )
        tree.fit()
        preds = tree.predict(X)
        for idx, p in enumerate(preds):
            self.assertEqual(y[idx], p, msg = "Prediction failed for index " + str(idx))

    def test_mushroom_dataset(self):
        X = mushrooms.drop("class", axis = 1); y = mushrooms["class"]
        tree = DecisionTree.Tree(data = X, labels = y)
        tree.fit()
        preds = tree.predict(X)
        for idx, p in enumerate(preds):
            self.assertEqual(y[idx], p, msg="Prediction failed for index " + str(idx))

    def test_prune_tree(self):
        X = mushrooms.drop("class", axis=1);
        y = mushrooms["class"]
        tree = DecisionTree.Tree(data=X, labels=y)
        tree.fit()
        tree.prune()
        preds = tree.predict(X)
        acc, precs, recall = accuracy_score(y, preds), \
                             precision_score(y, preds, pos_label = 'p'), \
                             recall_score(y, preds, pos_label = 'p')
        self.assertGreater(acc, 0.99)
        self.assertGreater(precs, 0.99)
        self.assertGreater(recall, 0.99)


if __name__ == "main":
    unittest.main()