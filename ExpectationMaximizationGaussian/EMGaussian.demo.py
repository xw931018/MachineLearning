from ExpectationMaximizationGaussian import EMGaussian
from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, n_features=2, centers = 2, cluster_std=1)

em_model = EMGaussian.EM_Gaussian(K = 2)
em_model.fit(X, iterations = 1000)
em_model.plot_center()