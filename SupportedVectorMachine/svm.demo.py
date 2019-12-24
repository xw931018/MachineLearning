import numpy as np
from sklearn.svm import SVC

from SupportedVectorMachine import svm

DIM = 2
COLORS = ['red', 'blue']

M1 = np.ones((DIM,))

M2 = 2 * np.ones((DIM,))

V1 = np.diag(0.4 * np.ones((DIM,)))

V2 = np.diag(0.4 * np.ones((DIM,)))

NUM = 100
if __name__ == "__main__":
    X1 = svm.gaussian_generator(M1, V1, NUM)
    y1 = np.ones((X1.shape[0],))
    X2 = svm.gaussian_generator(M2, V2, NUM)
    y2 = -np.ones((X2.shape[0],))
    x = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))

    svm.plot_data_with_labels(x, y, COLORS)


svc = SVC(kernel = "linear", C = 10)
svc.fit(x, y)

svm_lin = svm.SVM_Linear(x, y, 10)
alphas = svm_lin.dual_fit()
(w, b) = svm_lin.fit()

svm_lin.plot_2d()

print(svm_lin.w, svc.coef_)
print(svm_lin.b, svc.intercept_)

def RBF(x, y):
    return np.exp(-2*(x - y).dot(x-y))

svm_kernel = svm.SVM_Linear(x, y, 10, kernel = RBF)
alphas_kernel = svm_kernel.dual_fit()
svm_kernel.fit()

svm_kernel.plot_2d()

from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 200, noise = 0.15, random_state = 1)

y = (y - 0.5)*2

def polynomial(x, y):
    return (x.dot(y) + 5)**5

svm.plot_data_with_labels(X, y)

svm_kernel = svm.SVM_Linear(X, y, C = 10, kernel = RBF)

svm_kernel.fit(X, y)
svm_kernel.plot_2d()