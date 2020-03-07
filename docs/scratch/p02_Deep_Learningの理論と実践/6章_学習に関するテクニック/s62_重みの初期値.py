# ## 重みの初期値

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 6章2節
# の内容を学習しながら、重みの初期値の影響を調べます。

# # 6.2.2 隠れ層のアクティベーション分布

# 実験に必要な活性化関数などを定義します。
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def xavier(n):
    return np.sqrt(1 / n)


def he(n):
    return np.sqrt(2 / n)


# 隠れ層を活性化させるジェネレータと可視化関数を定義します。
import matplotlib.pyplot as plt  # isort:skip

input_data = np.random.randn(1000, 100)  # 1000個のデータ
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層


def evalute(activate, std):
    weight = np.random.randn(node_num, node_num) * std
    x = input_data
    for _ in range(hidden_layer_size):
        x = activate(x @ weight)
        yield x


def plot(activations, ylim=None):
    plt.figure(figsize=(12, 3))
    for i, a in enumerate(activations):
        plt.subplot(1, hidden_layer_size, i + 1)
        plt.title(str(i + 1) + "-layer")
        if i != 0:
            plt.yticks([], [])
        if ylim:
            plt.ylim(0, ylim)
        plt.hist(a.flatten(), 30, range=(0, 1))


# #Fig 6.10 <code>evalute(sigmoid, std=1)</code>
plot(evalute(sigmoid, std=1))

# #Fig 6.11 <code>evalute(sigmoid, std=0.01)</code>
plot(evalute(sigmoid, std=0.01))

# #Fig 6.13 <code>evalute(sigmoid, std=xavier(node_num))</code>
plot(evalute(sigmoid, std=xavier(node_num)))

# ### ReLUの場合の重みの初期値

# #Fig 6.14.1 <code>evalute(ReLU, std=0.01)</code>
plot(evalute(ReLU, std=0.01), ylim=7000)

# #Fig 6.14.2 <code>evalute(ReLU, std=xavier(node_num))</code>
plot(evalute(ReLU, std=xavier(node_num)), ylim=7000)

# #Fig 6.14.3 <code>evalute(ReLU, std=he(node_num))</code>
plot(evalute(ReLU, std=he(node_num)), ylim=7000)


# ### MNISTデータセットによる重みの初期値の比較

# `ch06/weight_init_compare.py`および`common/multi_layer_net.py`を参考にして、重みの初期
# 値が学習に与える影響について実験を行います。

# Ivoryライブラリでは、重みを正規分布で初期化するとき、標準偏差には数値のほかに
# 、XavierおよびHeの初期値を指定することもできます。
from ivory.layers.affine import Affine  # isort:skip

affine = Affine((100, 50))
affine.set_variables()
W = affine.W.variable
print("Xavier", W.init("xavier").std())  # np.sqrt(1/100) ≒ 0.10
print("He    ", W.init("he").std())  # np.sqrt(2/100) ≒ 0.14
# ここで注意することは、`Variable`インスタンスの`init`メソッドは、初期化された値を返しま
# すが、自分自身の値は更新しないことです。

# 以上の方法を使って、課題を準備します。
from ivory.core.model import sequential  # isort:skip
from ivory.core.optimizer import SGD  # isort:skip
from ivory.datasets.mnist import load_dataset  # isort:skip

data = load_dataset(train_only=True)
data.batch_size = 128
data.epochs = -1
data.random = True

net = [
    ("input", 784),
    (4, "affine", 100, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
model = sequential(net)
optimizer = SGD(learning_rate=0.01)
optimizer.set_model(model)


# 続いて訓練を行うジェネレータ関数を定義します。
def train(std):
    print(std)

    for p in model.weights:
        if p.name == "W":
            p.variable.data = p.variable.init(std)  # 明示的に値を代入する必要がある
        else:
            p.variable.data = p.variable.init()  # バイアス

    for _, (x, t) in zip(range(2000), data):
        model.set_data(x, t)
        model.forward()
        yield model.loss
        model.backward()
        optimizer.update()


# 訓練を実行します。
stds = [0.01, "xavier", "he"]
result = [list(train(std)) for std in stds]

# 可視化を行います。
from scipy.signal import savgol_filter  # isort:skip

markers = ["o", "s", "d"]
for k, x in enumerate(savgol_filter(result, 9, 3)):
    plt.plot(x, marker=markers[k], markevery=100, label=str(stds[k]))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.xlim(0, 2000)
plt.ylim(0, 2.5)
plt.legend()
plt.show()
