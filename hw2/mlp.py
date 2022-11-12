import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def relu(x):
    return np.maximum(0, x)


def relu_d(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


x = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

y = [
    0,
    1,
    1, 
    0
]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)


class Perceptron:
    def __init__(self, n_input, n_hidden, n_output, 
                rate=0.05, act=relu, act_d=relu_d, seed=1):
        self.rate = rate
        self.act = act
        self.act_d = act_d
        self.random_state = np.random.RandomState(seed)

        self.Wh = np.random.randn(n_input, n_hidden)
        self.Wo = np.random.randn(n_hidden, n_output)
        self.Bh = np.full((1, n_hidden), 0.1)
        self.Bo = np.full((1, n_output), 0.1)


    def fit(self, x, y, epochs):
        for epoch in range(epochs):
            for i in range(x.shape[0]):
                o = self.predict(x[i])
                Eo = (o - y) * self.act_d(self.Zo)
                Eh = Eo * self.Wo * self.act_d(self.Zh)

                dWo = Eo * self.H
                dWh = Eh * x[i]

                self.Wh -= self.rate * dWh
                self.Wo -= self.rate * dWo
                

    def predict(self, x):
        # Hidden layer
        self.Zh = np.dot(x, self.Wh) + self.Bh
        self.H = self.act(self.Zh)

        # Output layer
        self.Zo = np.dot(self.H, self.Wo) + self.Bo
        y = self.act(self.Zo)

        return y 


clf = Perceptron(2, 1, 1, act=relu, act_d=relu_d)
clf.fit(x, y, 500)

print(clf.predict([0, 1]))