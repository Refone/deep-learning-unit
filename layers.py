# Relu Layer
class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


# Affine Layer
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


# Softmax With Loss Layer
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        pass

    def backward(self, dout=1):
        pass
