import numpy as np
import pandas as pd
import seaborn as sns;

sns.set()
import copy
from matplotlib import pyplot as plt

initialization_methods = ["random", "furthest", "kmeans++"]


class KMeansPlus:
    def __init__(self, K):
        self._K = K

    def initializ_center(self, data, method="random", K=None):
        if K is None:
            K = self._K
        if method not in initialization_methods:
            raise ValueError("Please choose a method between" + str(initialization_methods))
        min_features, max_features = np.min(data, axis=0), np.max(data, axis=0)
        centers = np.array([np.random.uniform(min, max, size=K)
                            for min, max in zip(min_features, max_features)]).T
        if method is "random":
            return centers

        if method is "furthest":
            center = centers[0]  # Take the first random centroid
            centers = np.array([center])
            tmp = copy.deepcopy(data)
            for i in range(K - 1):  # Find the furthest point to previous centroids
                distances = np.array([np.linalg.norm(x - centers) for x in tmp])
                arg = np.argmax(distances)
                centers = np.append(centers, tmp[arg]).reshape(i + 2, 2)
                tmp = np.delete(tmp, arg, axis=0)
            return centers
            # K-means++ initialization

    def fit(self, data, labels, initial="kmeans++"):
        self._data = data
        self._labels = labels
        centroids = self.initialize_center(data, K=None)
