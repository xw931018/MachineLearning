import numpy as np
import pandas as pd
import operator
import copy
from collections import Counter

relu = lambda x: max([x, 0])
relu_def = lambda x: 1 * (x > 0)

logistic = lambda x: np.exp(x) / (1 + np.exp(x))
logistic_def = lambda x: np.exp(x) / (2 * np.exp(x) + np.exp(2 * x) + 1)


class Neuron:
    def __init__(self, activate=logistic, intercept=0,
                 activate_def=logistic_def, output_layer=False,
                 n_previous=None, layer=None, position=None, invalue=None):
        self._layer = layer
        self._position = position
        self._activate = activate
        self._activate_def = activate_def  # The derivative of activation function
        self._in_value = invalue
        self._activated_value = 0
        self._output_layer = output_layer
        self._category = None
        if invalue is not None:
            self._activated_value = self._activate(invalue)

    def activate(self, invalue):
        if invalue is not None:
            self._in_value = invalue
        self._activated_value = self._activate(self._in_value)

    def delta(self, delta):
        self._delta = delta

    @property
    def output(self):
        return self._activated_value


class Network:
    def __init__(self, data, labels, n_neurons=[],
                 activate=logistic,
                 activate_def=logistic_def):  # n_neurons are the number of neurons in the hidden layers
        self._activate = activate
        self._activate_def = activate_def
        self._data = data
        self._labels = labels
        self._counters = Counter(labels)
        self._n_category = len(self._counters.values())  # Number of neurons in the last layer
        self._n_features = data.shape[1]  # Number of neurons in the first layer
        self._n_samples = data.shape[0]
        n_hidden_layers = len(n_neurons)
        self.generate_network(n_hidden_layers, n_neurons)

    def generate_network(self, n_hidden_layers=0, n_neurons=[]):
        self._n_layers = 2 + n_hidden_layers
        self._shape = [self._n_features] + n_neurons + [self._n_category]
        self._weights = []
        self._intercepts = []  # initiate the weights and intercepts
        self._neurons = []
        self._neurons.append(np.array([Neuron(activate=lambda x: x, activate_def=lambda x: 1)
                                       for i in range(self._shape[0])]))  # Input layer
        for l in np.arange(self._n_layers - 1):
            self._intercepts.append(np.random.normal(size=self._shape[l + 1]))
            self._weights.append(np.random.normal(size=(self._shape[l + 1], self._shape[l])))
            self._neurons.append(np.array([Neuron(intercept=self._intercepts[l][i],
                                                  activate=self._activate,
                                                  activate_def=self._activate_def,
                                                  output_layer=l == self._n_layers - 2)
                                           for i in range(len(self._intercepts[l]))]))
            if l == self._n_layers - 2:
                for i in range(self._n_category):
                    self._neurons[l + 1][i]._category = list(self._counters.keys())[i]
        self._weights = np.array(self._weights)
        self._intercepts = np.array(self._intercepts)

        self._weights_grad = copy.deepcopy(self._weights)
        self._intercepts_grad = copy.deepcopy(self._intercepts)
        self._deltas = copy.deepcopy(self._intercepts_grad)

    def feed_forward_one_sample(self, x):
        for i in range(len(x)):
            self._neurons[0][i].activate(x[i])
        for l in range(self._n_layers - 1):
            pre_activated = np.array([self._neurons[l][i].output
                                      for i in range(self._shape[l])])
            inputs = self._weights[l].dot(pre_activated) + self._intercepts[l]
            for i in range(self._shape[l + 1]):
                self._neurons[l + 1][i].activate(inputs[i])  # Every neuron is activated

    def output_one_sample(self, x):
        self.feed_forward_one_sample(x)
        return dict(zip(np.array([neuron._category
                                  for neuron in self._neurons[self._n_layers - 1]]),
                        np.array([neuron.output
                                  for neuron in self._neurons[self._n_layers - 1]])))

    def predict_one_sample(self, x):
        output = self.output_one_sample(x)
        return max(output.items(), key=operator.itemgetter(1))[0]

    def loss_one_sample(self, x, y):  # Here for simplicity we first apply qudratic loss
        # instead of entropy loss
        output = self.output_one_sample(x)
        expected = dict(zip(self._counters.keys(), np.zeros(self._n_category)))
        expected[y] = 1
        loss = 0
        for key in expected.keys():
            loss += (expected[key] - output[key]) ** 2
        return 0.5 * loss

    def loss_train_data(self):
        loss = 0
        for i in range(self._n_samples):
            loss += self.loss_one_sample(self._data.loc[i], self._labels[i])
        return loss / self._n_samples

    def back_propagation_one_sample(self, x, y):
        L = self._n_layers - 1
        for i in range(self._shape[L]):
            neuron = self._neurons[L][i]
            self._deltas[L - 1][i] = (neuron._activated_value - (y == neuron._category)) * neuron._activate_def(
                neuron._in_value)
        self._intercepts_grad[L - 1] = self._deltas[L - 1]
        self._weights_grad[L - 1] = self._deltas[L - 1][:, None] * np.array(
            [neuron._activated_value for neuron in self._neurons[L - 1]])

        for l in np.arange(self._n_layers - 2, 0, -1):
            self._deltas[l - 1] = np.multiply(self._weights[l].T.dot(self._deltas[l]),
                                              np.array([neuron._activate_def(neuron._in_value) for neuron in
                                                        self._neurons[l]]))
            self._intercepts_grad[l - 1] = self._deltas[l - 1]
            self._weights_grad[l - 1] = self._deltas[l - 1][:, None] * np.array(
                [neuron._activated_value for neuron in self._neurons[l - 1]])

    def fit(self, step=0.001, iterations=2000):
        for i in range(iterations):
            dw = 0
            db = 0
            for n in range(self._n_samples):
                self.feed_forward_one_sample(self._data.loc[n])
                self.back_propagation_one_sample(self._data.loc[n], self._labels[n])
                dw = dw + self._weights_grad
                db = db + self._intercepts_grad

            dw = dw / self._n_samples
            db = db / self._n_samples
            self._weights = np.array(self._weights) - step * dw
            self._intercepts = np.array(self._intercepts) - step * db

