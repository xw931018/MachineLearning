from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

class EM_Gaussian:
    def __init__(self, K):
        self._K = K

    def initialize_params(self, data, K=None):
        if K is None:
            K = self._K
        partition = np.array_split(np.random.permutation(range(len(data))), K)
        p = [1 / K] * K  # initializing probabilities
        mu = []
        sigma = []
        for i in range(K):
            part = partition[i];
            cluster = data[part]
            mean = np.mean(cluster, axis=0)
            var = (cluster - mean).T.dot(cluster - mean)
            mu.append(mean)
            sigma.append(var)
        self._p = p
        self._mu = mu
        self._sigma = sigma
        return p, mu, sigma

    def fit(self, data, K=None, iterations=1000):
        if K is None:
            K = self._K
        self._data = data
        self._n_samples = data.shape[0]
        self._n_features = data.shape[1]
        self.initialize_params(data)
        p = self._p;
        mu = self._mu;
        sigma = self._sigma
        for iter in range(iterations):
            dist = [[p[k] * stats.multivariate_normal.pdf(x, mean=mu[k], cov=sigma[k])
                     for k in range(K)] for x in data]
            distribution = [d / np.sum(d) for d in dist]
            p = np.mean(distribution, axis=0)
            mu = [np.sum([data[i] * distribution[i][k] for i in range(self._n_samples)], axis=0)
                  / np.sum(distribution, axis=0)[k]
                  for k in range(K)]
            tmp_mu = np.array(mu)
            sigma = [np.sum(
                [(data[i] - tmp_mu[k]).reshape(-1, 1).dot((data[i] - tmp_mu[k]).reshape(1, -1)) * distribution[i][k]
                 for i in range(self._n_samples)], axis=0)
                     / np.sum(distribution, axis=0)[k]
                     for k in range(K)]
        self._p = p;
        self._mu = mu;
        self._sigma = sigma
        return p, mu, sigma

    def predict_one_sample(self, x):
        """Calculate the normalized distances from the point x to cluster centers and choose the closest"""
        distances = np.linalg.norm(x - self._mu, axis=1) / np.sqrt(np.linalg.det(self._sigma))
        label = np.argmin(distances)
        return label

    def predict(self, data):
        return np.array([self.predict_one_sample(point) for point in data])

    def plot_center(self):
        centers = np.array(self._mu)
        plt.scatter(self._data[:, 0], self._data[:, 1])
        plt.scatter(centers[:, 0], centers[:, 1], color='r')
        plt.show()