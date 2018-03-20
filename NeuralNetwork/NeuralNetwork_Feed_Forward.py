import numpy as np
import pandas as pd
import operator
from collections import Counter


class Neuron:
    def __init__(self, activate=lambda x: 1 / (1 + np.exp(-x)), intercept=0,
                 activate_def=lambda x: 1 / (2 + np.exp(x) + np.exp(-x)),
                 n_previous=None, layer=None, position=None, invalue=None):
        self._layer = layer
        self._position = position
        self._activate = activate
        self._activate_def = activate_def  # The derivative of activation function
        self._in_value = invalue
        self._activated_value = 0
        if invalue is not None:
            self._activated_value = self._activate(invalue)

    def activate(self, invalue):
        if invalue is not None:
            self._in_value = invalue
        self._activated_value = self._activate(self._in_value)

    @property
    def output(self):
        return self._activated_value


class Network:
    def __init__(self, data, labels, n_hidden_layers=0,
                 n_neurons=[]):  # n_neurons are the number of neurons in the hidden layers
        self._data = data
        self._labels = labels
        self._counters = Counter(labels)
        self._n_category = len(self._counters.values())  # Number of neurons in the last layer
        self._n_features = data.shape[1]  # Number of neurons in the first layer
        self._n_samples = data.shape[0]
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
            self._neurons.append(np.array([Neuron(intercept=self._intercepts[l][i])
                                           for i in range(len(self._intercepts[l]))]))

    def feed_forward_one_sample(self, x):
        for i in range(len(x)):
            self._neurons[0][i].activate(x[i])
        for l in range(self._n_layers - 1):
            pre_activated = np.array([self._neurons[l][i].output
                                      for i in range(self._shape[l])])
            inputs = self._weights[l].dot(pre_activated) + self._intercepts[l]
            for i in range(self._shape[l + 1]):
                self._neurons[l + 1][i].activate(inputs[i])

    def output_one_sample(self, x):
        self.feed_forward_one_sample(x)
        return dict(zip(self._counters.keys(),
                        np.array([neuron.output
                                  for neuron in self._neurons[self._n_layers - 1]])))

    def predict_one_sample(self, x):
        output = self.output_one_sample(x)
        return max(output.items(), key=operator.itemgetter(1))[0]

    def loss_one_sample(self, x, y):  # Here for simplicity we first apply qudratic loss
        # instead of entropy loss
        predict = self.predict_one_sample(x)
        return 0.5 * (predict - y) ** 2

    def loss_train_data(self):
        loss = 0
        for i in range(self._n_samples):
            loss += self.loss_one_sample(self._data.loc[i], self._labels[i])
        return loss / self._n_samples

    def back_propagation_one_sample(self, x):
        pass

    def fit(self):
        pass
