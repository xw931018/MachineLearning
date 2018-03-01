import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def gaussian_generator(m, c, num):
    return np.random.multivariate_normal(m, c, num)


def plot_data_with_labels(x, y, COLORS=["red", "blue"], w=None, b=None, supt_vects=None, kernel=None, alphas=None):
    labels = np.unique(y)
    NN = 30
    for i in range(len(labels)):
        x_class = x[y == labels[i]]
        plt.scatter(x_class[:, 0], x_class[:, 1], c=COLORS[i])
    x_min, x_max = np.min(x[:, 0]), np.max(x[:, 0])
    x_range = np.linspace(x_min - 0.5, x_max + 0.5, NN)
    y_min, y_max = np.min(x[:, 1]), np.max(x[:, 1])
    y_range = np.linspace(y_min - 0.5, y_max + 0.5, NN)
    X, Y = np.meshgrid(x_range, y_range)
    xx = np.c_[X.ravel(), Y.ravel()]
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    if w is not None and b is not None:
        t = np.linspace(x_min, x_max, 100)
        intercept = -b[0] / w.flatten()[1]
        slope = -w.flatten()[0] / w.flatten()[1]
        plt.plot(t, slope * t + intercept, 'k-')

        if len(w.flatten()) == 2:
            zz = xx.dot(w.flatten()) + b
            zz = zz.reshape(X.shape)
            ct = plt.contourf(x_range, y_range, zz, cmap=cm, alpha=.7)
            plt.colorbar(ct)
    if kernel is not None and alphas is not None:
        zz = np.zeros(len(xx))
        print(zz.shape)
        for i, point in enumerate(xx):
            DD = np.diag([kernel(x[i, :], point) for i in np.arange(x.shape[0])])
            yy = y * alphas
            zz[i] = sum(DD.dot(yy.T)) + b
        zz = zz.reshape(X.shape)
        ct = plt.contourf(x_range, y_range, zz, cmap=cm, alpha=.7)
        plt.colorbar(ct)

    if supt_vects is not None:
        plt.scatter(supt_vects[:, 0], supt_vects[:, 1], c="k", marker='^')
    plt.show()

import pandas as pd
import sys

from cvxopt import solvers, matrix


class SVM_Linear:
    # A linear SVM class that builds a classification model using linear SVM
    w = 0
    b = 0
    alphas = 0
    C = float('inf')
    supt_vects = 0

    def __init__(self, X, y, C=None, kernel=None):
        if not (X.shape[0] == y.shape[0]):
            sys.exit("Dimension of data not matching")
        self.__X = X
        self.__y = y
        if C is not None:
            self.C = C
        self.use_kernel = False
        if kernel is not None:
            self.kernel = kernel
            self.use_kernel = True

    def X(self):
        return self.__X

    def y(self):
        return self.__y

    def dual_fit(self, X=None, y=None, C=None):
        if X is None:
            X = self.__X
        else:
            self.__X = X
        if y is None:
            y = self.__y
        else:
            self.__y = y
        if C is None:
            C = self.C
        NUM = X.shape[0]
        DIM = X.shape[1]
        if not self.use_kernel:
            K = y[:, None] * X
            Q = matrix(K.dot(K.T))
        else:
            Q = np.zeros((NUM, NUM))
            for i in range(NUM):
                for j in range(NUM):
                    Q[i, j] = self.kernel(X[i, :], X[j, :])
            Q = matrix(Q)
        P = matrix(- np.ones((NUM, 1)))
        # When C is not infinity, extra rows should be added into G and h
        G = matrix(- np.eye(NUM))
        h = matrix(np.zeros(NUM))
        if self.C < float('inf'):
            G = matrix(np.concatenate((-np.eye(NUM), np.eye(NUM))))
            h = matrix(np.concatenate((np.zeros(NUM), C * np.ones(NUM))))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, P, G, h, A, b)
        alphas = np.array(sol['x']).T
        self.alphas = alphas
        return alphas

    def fit(self, X=None, y=None):
        if X is None:
            X = self.__X
        else:
            self.__X = X
        if y is None:
            y = self.__y
        else:
            self.__y = y
        alphas = self.dual_fit(X, y)
        if not self.use_kernel:
            w = X.T.dot((y * alphas).T).T
        else:
            print("Not use kernel")
            w = "w is not available with non-linear kernel"
        self.w = w
        spt_index = np.where(alphas > 1e-4)[1]
        self.supt_vects = X[spt_index, :]  # supporting vectors. The same with / without s_i
        p = X[spt_index[0], :]
        if not self.use_kernel:
            b = y[spt_index[0]] - w.dot(p)
        else:
            DD = np.diag([self.kernel(X[i, :], p) for i in np.arange(X.shape[0])])
            yy = y * alphas
            b = y[spt_index[0]] - sum(DD.dot(yy.T))
        if self.C < float('inf'):
            beta = self.C - alphas
            spt_index = np.where((beta > 1e-4) & (alphas > 1e-4))[1]
            print("Spt Index", spt_index)
            p = X[spt_index[0], :]
            if not self.use_kernel:
                b = y[spt_index[0]] - w.dot(p)
            else:
                DD = np.diag([self.kernel(X[i, :], p) for i in np.arange(X.shape[0])])
                yy = y * alphas
                b = y[spt_index[0]] - sum(DD.dot(yy.T))
            print(":gb", spt_index[0])
        self.b = b
        return (w, b)

    def predict(self, x):
        if not self.use_kernel:
            return x.dot(self.w.flatten()) + self.b
        DD = np.diag([self.kernel(self.__X[i, :], x) for i in np.arange(self.__X.shape[0])])
        yy = y * alphas
        return sum(DD.dot(yy.T)) + self.b

    def classify(self, x):
        if not self.use_kernel:
            return np.sign(self.predict(x))
        return np.sign(self.predict(x))

    def plot_2d(self, X=None, y=None):
        if X is None:
            X = self.__X
        if y is None:
            y = self.__y
        if not self.use_kernel:
            plot_data_with_labels(X, y, ['red', 'blue'], self.w, self.b, self.supt_vects)
        else:
            plot_data_with_labels(X, y, ['red', 'blue'], b=self.b, kernel=self.kernel, alphas=self.alphas)