import numpy as np
import copy

initialization_methods = ["random", "furthest", "kmeans++"]


class KMeansPlus:
    def __init__(self, K):
        self._K = K

    def initialize_center(self, data, method="random", K=None):
        if K is None:
            K = self._K
        n_samples = data.shape[0]
        n_features = data.shape[1]
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
                centers = np.append(centers, tmp[arg]).reshape(-1, n_features)
                tmp = np.delete(tmp, arg, axis=0)
            return centers
        # K-means++ initialization
        center = centers[0]
        centers = np.array([center])
        for i in range(K - 1):
            min_centroid_distances = np.array([np.min(np.linalg.norm(x - centers, axis=1))
                                               for x in data])
            proba = min_centroid_distances / np.sum(min_centroid_distances)
            index = np.random.choice(range(data.shape[0]), p=proba)
            new_centroid = data[index]
            centers = np.append(centers, new_centroid).reshape(-1, n_features)
        return centers

    def _closer(self, sample, centroids, k):  # to judge if a sample point is closer to the kth centroid
        # than to other centroids
        distances = np.linalg.norm(sample - centroids, axis=1)
        argmin = np.argmin(distances)
        return k == argmin

    def fit(self, data, iterations=200, K=None, initial_method="kmeans++"):
        if K is None:
            K = self._K
        self._data = data
        self._n_samples = data.shape[0]
        self._n_features = data.shape[1]
        centroids = self.initialize_center(data, method=initial_method, K=K)
        for iter in range(iterations):
            centroids_candidates = [[] for i in range(K)]
            tmp = copy.deepcopy(data)
            for i in range(K):
                indexes_to_delete = []
                for index, x in enumerate(tmp):
                    if self._closer(x, centroids, i):
                        centroids_candidates[i].append(x)
                        indexes_to_delete.append(index)
                tmp = np.delete(tmp, index, axis=0)
            centroids = np.array([np.mean(np.array(cluster), axis=0) for cluster in centroids_candidates])
        self._centroids = centroids
        classified = []
        for x in data:
            classified.append(self.predict_one_sample(x))
        self._classified = classified
        return centroids, classified

    def predict_one_sample(self, x):
        return np.argmin(np.linalg.norm(x - self._centroids, axis=1))

    def predict(self, data):
        return [self.predict_one_sample(sample) for sample in data]