import numpy as np


def softmax(x):
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return x / x.sum(axis=1, keepdims=True)


def cross_entropy(y, t):
    return -np.log(y[np.arange(len(t)), t] + 1e-7).mean()
