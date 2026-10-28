import numpy as np

from layers import Affine, Relu, SoftmaxWithLoss


# 按 dims 组装网络，如 NN([2, 32, 32, 3]) 即 2 -> 32 -> 32 -> 3
class NN:
    def __init__(self, dims):
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(dims, dims[1:])):
            W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)  # He 初始化
            self.layers.append(Affine(W, np.zeros(n_out)))
            if i < len(dims) - 2:
                self.layers.append(Relu())
        self.criterion = SoftmaxWithLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.criterion.forward(self.forward(x), t)

    def backward(self):
        dout = self.criterion.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def predict(self, x):
        return self.forward(x).argmax(axis=1)
