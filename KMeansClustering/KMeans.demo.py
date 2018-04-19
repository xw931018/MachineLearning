import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from KMeansClustering import KMeans as km

X, y_true = make_blobs(n_samples = 300, n_features = 2, centers = 4, cluster_std = 1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

km_model = km.KMeansPlus(K = 4)


cc = km_model.fit(data = X, iterations = 100)
plt.scatter(X[:, 0], X[:, 1], c = cc[1])
plt.scatter(cc[0][:, 0], cc[0][:, 1], color = 'r')
plt.show()
