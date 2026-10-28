import numpy as np

from nn_net import NN
from optimizer import AdamW


def moons(n):
    t = np.linspace(0, np.pi, n // 2)
    x = np.r_[np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]]
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    return x + 0.1 * np.random.randn(n, 2), y


np.random.seed(0)
x, y = moons(800)

net = NN([2, 32, 32, 2])
opt = AdamW(net.layers, lr=1e-2, wd=1e-2)

for epoch in range(1001):
    loss = net.loss(x, y)  # 前向传播 + 计算 loss
    net.backward()         # 反向传播
    opt.step()             # AdamW 更新

    if epoch % 200 == 0:
        acc = (net.predict(x) == y).mean()
        print(f"epoch {epoch:4d} | loss {loss:.4f} | acc {acc:.3f}")
