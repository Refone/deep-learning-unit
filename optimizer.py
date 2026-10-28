import numpy as np


class AdamW:
    def __init__(self, layers, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, wd=1e-2):
        self.layers = layers
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.t = 0
        self.params = [p for layer in layers for p in layer.params]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        grads = [g for layer in self.layers for g in layer.grads]
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1 / (np.sqrt(v / c2) + self.eps) + self.wd * p)
