# Path: models/ann_numpy.py

import numpy as np

class ANN:
    def __init__(self, layer_dims, learning_rate=0.01, epochs=1000, seed=42):
        """
        layer_dims: list of ints, e.g., [12, 32, 16, 1]
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameters = {}
        self.losses = []
        self.rng = np.random.default_rng(seed)
        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization for ReLU layers, Xavier for sigmoid output
        L = len(self.layer_dims)
        for l in range(1, L):
            fan_in = self.layer_dims[l-1]
            fan_out = self.layer_dims[l]
            if l < L - 1:  # hidden layers (ReLU)
                scale = np.sqrt(2.0 / fan_in)
            else:          # output layer (sigmoid)
                scale = np.sqrt(1.0 / fan_in)
            self.parameters['W' + str(l)] = self.rng.normal(0.0, scale, size=(fan_out, fan_in))
            self.parameters['b' + str(l)] = np.zeros((fan_out, 1))

    @staticmethod
    def _sigmoid(Z):
        return 1.0 / (1.0 + np.exp(-Z))

    @staticmethod
    def _relu(Z):
        return np.maximum(0.0, Z)

    @staticmethod
    def _relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def _sigmoid_derivative(A):
        return A * (1.0 - A)

    def _forward_propagation(self, X):
        # X: (m, n_features)
        A = X.T  # (n_features, m)
        cache = {'A0': A}
        L = len(self.layer_dims) - 1

        for l in range(1, L):
            Z = self.parameters['W' + str(l)] @ cache['A' + str(l-1)] + self.parameters['b' + str(l)]
            A = self._relu(Z)
            cache['Z' + str(l)] = Z
            cache['A' + str(l)] = A

        ZL = self.parameters['W' + str(L)] @ cache['A' + str(L-1)] + self.parameters['b' + str(L)]
        AL = self._sigmoid(ZL)
        cache['Z' + str(L)] = ZL
        cache['A' + str(L)] = AL
        return AL, cache

    @staticmethod
    def _compute_cost(AL, Y):
        # AL: (1, m), Y: (m, 1)
        m = Y.shape[0]
        YT = Y.T
        eps = 1e-8
        cost = -np.sum(YT * np.log(AL + eps) + (1 - YT) * np.log(1 - AL + eps)) / m
        return float(cost)

    def _backward_propagation(self, cache, X, Y):
        grads = {}
        m = X.shape[0]
        L = len(self.layer_dims) - 1
        YT = Y.T

        AL = cache['A' + str(L)]
        dAL = -(np.divide(YT, AL + 1e-8) - np.divide(1 - YT, 1 - AL + 1e-8))
        dZL = dAL * self._sigmoid_derivative(AL)
        grads['dW' + str(L)] = (1.0 / m) * (dZL @ cache['A' + str(L-1)].T)
        grads['db' + str(L)] = (1.0 / m) * np.sum(dZL, axis=1, keepdims=True)

        dZ_next = dZL
        for l in reversed(range(1, L)):
            dA = self.parameters['W' + str(l+1)].T @ dZ_next
            dZ = dA * self._relu_derivative(cache['Z' + str(l)])
            grads['dW' + str(l)] = (1.0 / m) * (dZ @ cache['A' + str(l-1)].T)
            grads['db' + str(l)] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ_next = dZ

        return grads

    def _update_parameters(self, grads):
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            self.parameters['W' + str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]

    def fit(self, X, Y, verbose_every=100):
        for epoch in range(self.epochs):
            AL, cache = self._forward_propagation(X)
            cost = self._compute_cost(AL, Y)
            grads = self._backward_propagation(cache, X, Y)
            self._update_parameters(grads)

            if (epoch % verbose_every) == 0:
                self.losses.append(cost)
                print(f"Epoch {epoch:4d} | Loss: {cost:.4f}")

    def predict_proba(self, X):
        AL, _ = self._forward_propagation(X)
        return AL.T  # (m, 1)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
