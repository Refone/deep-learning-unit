import numpy as np

from functions import cross_entropy, softmax


# 所有层的统一接口；可训练层通过 params / grads 暴露参数与梯度
class Layer:
    params = []
    grads = []


# Relu Layer
class Relu(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


# Affine Layer
class Affine(Layer):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.grads[0][...] = self.x.T @ dout  # 原地写入，保持梯度数组引用稳定
        self.grads[1][...] = dout.sum(axis=0)
        return dout @ self.W.T


# Softmax With Loss Layer
class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, t)
        return self.loss

    def backward(self, dout=1):
        d = self.y.copy()
        d[np.arange(len(self.t)), self.t] -= 1
        return dout * d / len(self.t)
