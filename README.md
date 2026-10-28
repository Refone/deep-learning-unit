# 极简深度学习框架（NumPy）

每层实现 `forward` / `backward`，网络由维度列表组合，优化器为 AdamW。以下只给出代码与数学公式的对应关系。

```bash
python nn_demo.py
```

```python
net = NN([2, 32, 32, 2])
opt = AdamW(net.layers, lr=1e-2, wd=1e-2)
loss = net.loss(x, y)   # 前向 + 交叉熵
net.backward()          # 写回各层梯度
opt.step()              # AdamW 更新
```

## 符号

| 符号 | 含义 |
| --- | --- |
| $N$ | batch 大小 |
| $C$ | 类别数 |
| $X$ | 输入，$N\times d_{in}$ |
| $W$ | 权重，$d_{in}\times d_{out}$ |
| $b$ | 偏置，$d_{out}$ |
| $Y$ | 层输出；$Z$ 为 logits |
| $t$ | 整数标签，$t_i\in\{0,\dots,C-1\}$ |
| $T$ | 标签的 one-hot 矩阵 |
| $\frac{\partial L}{\partial Y}$ | 上游梯度，代码中为 `dout` |

$\mathbf 1_N$ 表示长度为 $N$ 的全 1 向量。

## 1. functions.py

```python
def softmax(x):
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return x / x.sum(axis=1, keepdims=True)

def cross_entropy(y, t):
    return -np.log(y[np.arange(len(t)), t] + 1e-7).mean()
```

$$
\operatorname{softmax}(x)_{ij}
=\frac{\exp(x_{ij}-m_i)}{\sum_{k=1}^{C}\exp(x_{ik}-m_i)},
\qquad m_i=\max_k x_{ik}
$$

$$
L=-\frac{1}{N}\sum_{i=1}^{N}
\log\left(Y_{i,t_i}+10^{-7}\right)
$$

行最大值减法与 $10^{-7}$ 均为数值稳定项。

## 2. layers.py

统一接口：`forward` 返回 $Y=f(X)$；`backward(dout)` 接收 $\frac{\partial L}{\partial Y}$，返回 $\frac{\partial L}{\partial X}$，并写入参数梯度。

$$
\frac{\partial L}{\partial X}
=J_f(X)^T\frac{\partial L}{\partial Y}
$$

### Affine

```python
def forward(self, x):
    self.x = x
    return x @ self.W + self.b

def backward(self, dout):
    self.grads[0][...] = self.x.T @ dout
    self.grads[1][...] = dout.sum(axis=0)
    return dout @ self.W.T
```

$$
Y=XW+\mathbf 1_N b^T
$$

$$
\frac{\partial L}{\partial W}
=X^T\frac{\partial L}{\partial Y},
\qquad
\frac{\partial L}{\partial b}
=\mathbf 1_N^T\frac{\partial L}{\partial Y},
\qquad
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial Y}W^T
$$

### ReLU

```python
def forward(self, x):
    self.mask = x > 0
    return x * self.mask

def backward(self, dout):
    return dout * self.mask
```

$$
Y=\max(0,X)
$$

$$
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial Y}\odot\mathbf 1[X>0]
$$

### SoftmaxWithLoss

```python
def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy(self.y, t)
    return self.loss

def backward(self, dout=1):
    d = self.y.copy()
    d[np.arange(len(self.t)), self.t] -= 1
    return dout * d / len(self.t)
```

$$
Y=\operatorname{softmax}(Z),
\qquad
L=-\frac{1}{N}\sum_{i=1}^{N}
\log\left(Y_{i,t_i}+10^{-7}\right)
$$

$$
\frac{\partial L}{\partial Z}
=\text{dout}\cdot\frac{Y-T}{N}
$$

## 3. nn_net.py

网络由维度列表定义：

$$
\mathcal D=(d_0,d_1,\dots,d_L)
$$

每对相邻维度对应一层 Affine；除最后一层外，每层后接 ReLU。

### 前向

$$
H_0=X
$$

$$
A_l=H_{l-1}W_l+\mathbf 1_N b_l^T,
\qquad
H_l=\operatorname{ReLU}(A_l)\quad(l<L)
$$

$$
Z=A_L,
\qquad
Y=\operatorname{softmax}(Z)
$$

### 初始化

$$
W_l\sim\mathcal N\left(0,\frac{2}{d_{l-1}}\right),
\qquad b_l=0
$$

### 反向

```python
def backward(self):
    dout = self.criterion.backward()
    for layer in reversed(self.layers):
        dout = layer.backward(dout)
```

$$
\frac{\partial L}{\partial A_L}
=\frac{Y-T}{N}
$$

$$
\frac{\partial L}{\partial W_l}
=H_{l-1}^T\frac{\partial L}{\partial A_l},
\qquad
\frac{\partial L}{\partial b_l}
=\mathbf 1_N^T\frac{\partial L}{\partial A_l}
$$

$$
\frac{\partial L}{\partial A_{l-1}}
=\left(
\frac{\partial L}{\partial A_l}W_l^T
\right)\odot\mathbf 1[A_{l-1}>0],
\qquad l>1
$$

对 $l=1$，输入梯度为：

$$
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial A_1}W_1^T
$$

`reversed(self.layers)` 依次执行上述 Affine / ReLU 反向。

## 4. optimizer.py

```python
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
```

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$$
p_t=p_{t-1}-\eta\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda p_{t-1}
\right)
$$

其中

$$
g_t=\frac{\partial L}{\partial p}
$$

`m`、`v` 初始为 0，`t` 初始为 0 且每次 `step` 先加 1，因此偏差修正使用 $t=1,2,\dots$；`lr`、`wd` 对应 $\eta$、$\lambda$。代码将自适应步长与解耦权重衰减 $\lambda p_{t-1}$ 合并为一行，并作用于所有参数。

## 5. nn_demo.py

two moons 数据：

$$
x_i^{(0)}=(\cos\theta_i,\sin\theta_i)+\varepsilon_i,
\qquad
x_i^{(1)}=(1-\cos\theta_i,\;0.5-\sin\theta_i)+\varepsilon_i
$$

其中 $\theta_i\in[0,\pi]$，$\varepsilon_i\sim\mathcal N(0,0.1^2I_2)$。

训练配置：

```python
net = NN([2, 32, 32, 2])
opt = AdamW(net.layers, lr=1e-2, wd=1e-2)
```

训练循环：

$$
\text{每一步：}\quad
L=\mathcal L(X,t),\qquad
\frac{\partial L}{\partial \theta}\leftarrow\text{backward}(),\qquad
\theta\leftarrow\text{AdamW.step}()
$$

训练完成后，loss 从约 $0.445$ 降到约 $4\times10^{-5}$，训练准确率约 100%。

## 6. 约定

- 梯度在每次 `backward()` 时覆盖，不做累加，因此无需 `zero_grad`。
- 标签为整数类别索引；$T$ 仅用于公式表达。
- `math.py` 因遮蔽标准库 `math`，已重命名为 `functions.py`。
- 代码不处理异常，默认输入为二维 batch。
