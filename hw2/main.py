import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 2/(1+np.exp(-x)) - 1


def sigmoid_d(x):
    return 1/2 * (1 - x**2)


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1 - (x ** 2)


def step(x):
    return np.heaviside(x)


def step_d(x):
    return NotImplemented


def relu(x):
    return max(0, x)


def relu_d(x):
    return int(x >= 0)


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
    def __init__(self, rate=0.05, min_error=0.01, act=sigmoid, act_d=sigmoid_d, seed=1, scale=1):
        # learning_rate
        self.rate = rate
        # activation function
        self.act = act
        # derivative of the activation function
        self.act_d = act_d
        # minimum error
        self.min_error = min_error
        # random state
        self.random_state = np.random.RandomState(seed)
        # scale for weight and bias
        self.scale = scale

    def fit(self, X, y, epochs, export='plot.png'):
        # number of features
        n = X.shape[0]
        # initial bias
        self.b = self.random_state.random() / self.scale
        # initial weights
        self.w = self.random_state.random((1, X.shape[1])) / self.scale
        print(self.w)
        errors = []
        MSEs = []

        for epoch in range(epochs):
            error = 0
            mse = 0
            for i in range(n):
                o = self.predict(X[i])
                self.w = self.w + self.rate * (y[i] - o) * self.act_d(o) * X[i]
                self.b = self.b + self.rate * (y[i] - o) * self.act_d(o)

                error += 1/2 * (y[i] - o)**2
                mse += (y[i] - o)**2

            errors.append(error)
            MSEs.append(mse / n)
            print(f'> {epoch}\t{error}')

            if error < self.min_error:
                break
        fig, ax = plt.subplots(2, figsize=(10, 10))

        ax[0].plot(errors)
        ax[1].plot(MSEs)

        ax[0].set_ylabel("loss error")
        ax[0].set_xlabel("epoch")
        ax[1].set_ylabel("MSE")
        ax[1].set_xlabel("epoch")

        fig.savefig(export)

    def predict(self, x):
        return self.act(np.dot(self.w, x) + self.b)


clf = Perceptron(act=relu, act_d=relu_d)
clf.fit(x, y, 500, export=f'mse.png')

print(clf.predict([0, 1]))