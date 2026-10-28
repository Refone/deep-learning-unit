"""在 MNIST 上训练手搓 MLP，并预测 var/sample_image.png。"""

import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from nn_net import NN
from optimizer import AdamW

ROOT = Path(__file__).resolve().parent
VAR = ROOT / "var"
DATA_DIR = VAR / "mnist"
CACHE = VAR / "mnist.npz"
SAMPLE = VAR / "sample_image.png"

# 数据集页面：https://platform.ultralytics.com/ultralytics/datasets/mnist
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/mnist.zip"
IMAGE_SIZE = 28


def _load_split(split_dir):
    xs, ys = [], []
    for digit in range(10):
        for path in (split_dir / str(digit)).glob("*.png"):
            with Image.open(path) as image:
                xs.append(np.asarray(image.convert("L"), dtype=np.float32).reshape(-1))
            ys.append(digit)
    return np.stack(xs) / 255.0, np.asarray(ys, dtype=np.int64)


def data_prepare():
    VAR.mkdir(exist_ok=True)
    if CACHE.exists():
        with np.load(CACHE) as data:
            train_data = data["x_train"], data["y_train"]
            test_data = data["x_test"], data["y_test"]
    else:
        if not DATA_DIR.exists():
            zip_path = VAR / "mnist.zip"
            print("downloading MNIST ...")
            urllib.request.urlretrieve(URL, str(zip_path))
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(VAR)
            zip_path.unlink()

        print("loading MNIST images ...")
        train_data = _load_split(DATA_DIR / "train")
        test_data = _load_split(DATA_DIR / "test")
        # 缓存为 npz，避免每次重新解码 7 万张 PNG
        np.savez_compressed(
            CACHE,
            x_train=train_data[0], y_train=train_data[1],
            x_test=test_data[0], y_test=test_data[1],
        )

    if not SAMPLE.exists():
        index = int(np.flatnonzero(test_data[1] == 7)[0])
        pixels = (test_data[0][index].reshape(IMAGE_SIZE, IMAGE_SIZE) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(SAMPLE)

    print(f"train {train_data[0].shape}, test {test_data[0].shape}, sample {SAMPLE}")
    return train_data, test_data


def build_nn_net(dims):
    return NN(dims)


def train(nn_net, train_data, epochs=10, batch_size=256, lr=1e-3):
    x, y = train_data
    optimizer = AdamW(nn_net.layers, lr=lr, wd=1e-2)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(x))
        total_loss = 0.0
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            loss = nn_net.loss(x[batch], y[batch])
            nn_net.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)
        print(f"epoch {epoch:2d}/{epochs}  loss {total_loss / len(x):.4f}")


def eval(nn_net, test_data, batch_size=1000):
    x, y = test_data
    correct = 0
    for start in range(0, len(x), batch_size):
        batch = slice(start, start + batch_size)
        correct += int((nn_net.predict(x[batch]) == y[batch]).sum())
    accuracy = correct / len(y)
    print(f"test accuracy: {accuracy:.4f}")
    return accuracy


def predict(nn_net, image_url):
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(image_url) as response:
            image = Image.open(BytesIO(response.read()))
    else:
        image = Image.open(image_url)

    with image:
        x = np.asarray(
            image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)),
            dtype=np.float32,
        ).reshape(1, -1) / 255.0
    if x.mean() > 0.5:
        x = 1.0 - x  # 浅色背景的输入转成 MNIST 的白字黑底

    digit = int(nn_net.predict(x)[0])
    print(f"prediction: {digit}")
    return digit


if __name__ == "__main__":
    np.random.seed(0)
    train_data, test_data = data_prepare()
    nn_net = build_nn_net([784, 256, 128, 10])
    train(nn_net, train_data)
    eval(nn_net, test_data)
    predict(nn_net, str(SAMPLE))
    print("Prediction completed.")
