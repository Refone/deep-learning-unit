from pathlib import Path

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


def save_nn_net(nn_net, path):
    """保存网络结构（dims）和全部参数到 .npz 文件。"""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    affines = [layer for layer in nn_net.layers if isinstance(layer, Affine)]
    dims = [affines[0].W.shape[0]] + [layer.W.shape[1] for layer in affines]
    params = [p for layer in nn_net.layers for p in layer.params]
    np.savez_compressed(
        path,
        dims=np.asarray(dims),
        **{f"param_{i}": p for i, p in enumerate(params)},
    )
    return path


def load_nn_net(path):
    """从 .npz 文件恢复网络结构和全部参数。"""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")

    with np.load(path) as data:
        dims = data["dims"].tolist()
        keys = sorted(
            (key for key in data.files if key.startswith("param_")),
            key=lambda key: int(key.split("_")[1]),
        )
        params = [data[key] for key in keys]

    nn_net = NN(dims)
    index = 0
    for layer in nn_net.layers:
        for param in layer.params:
            param[...] = params[index]
            index += 1
    return nn_net
