# # 6.1 パラメータの更新

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 6章1節
# の内容を学習しながら、Ivoryライブラリで`optimizer`モジュールを作成していきます。オプテ
# ィマイザーの基底クラスである`Optimizer`クラスは、学習するべきパラメータのリス
# ト`params`と学習率を属性に持ちます。（なお、`Optimizer`クラスおよびそのサブクラス
# はPython 3.7で導入された[データクラ
# ス](https://docs.python.org/ja/3/library/dataclasses.html)ですが、以下の定義で
# は`@dataclass`の記述は省略されています。）

# -hide
from ivory.utils.latex import Matrix

W = Matrix("w")
pW = W.spartial("L")

# {{ from ivory.core.optimizer import Optimizer }}
# ##Code <code>Optimizer</code>クラス
# {{ Optimizer # inspect }}

# 以下に、各々の更新手法の計算式と「ゼロから作るDeep Learning」を参考にして実装し
# た`Optimizer`のサブクラスをまとめます。

# # 6.1.2 SGD

# $${{W}} \leftarrow {{W}} - \eta{{pW}} $$

# {{ from ivory.core.optimizer import SGD }}
# ##Code <code>SGD</code>クラス
# {{ SGD # inspect }}

# # 6.1.4 Momentum

# $$\mathbf{v} \leftarrow \alpha\mathbf{v} - \eta{{pW}} $$

# $${{W}} \leftarrow {{W}} + \mathbf{v}$$

# {{ from ivory.core.optimizer import Momentum }}
# ##Code <code>Momentum</code>クラス
# {{ Momentum # inspect }}

# # 6.1.5 AdaGrad

# $$\mathbf{h} \leftarrow \mathbf{h} + {{pW}}\odot{{pW}}$$

# $${{W}} \leftarrow {{W}} - \eta\frac1{\sqrt{\mathbf{h}}}{{pW}}$$

# {{ from ivory.core.optimizer import AdaGrad }}
# ##Code <code>AdaGrad</code>クラス
# {{ AdaGrad # inspect }}

# # 6.1.6 Adam

# {{ from ivory.core.optimizer import Adam }}
# ##Code <code>Adam</code>クラス
# {{ Adam # inspect }}

# # 6.1.7 どの更新手法を用いるか？

# `ch06/optimizer_compare_naive.py`を参考にして、上述の更新手法を比較してみます。例題と
# なる関数の定義です。

import numpy as np  # isort:skip


def f(x):
    return x[0] ** 2 / 20 + x[1] ** 2


def df(x):
    return np.array([x[0] / 10, 2 * x[1]])


# オプティマイザーを引数としてとり、パラメータの更新ごとに結果をイールドするジェネレータ
# を定義します。例題の関数に合わせて2次元ベクトルのパラメータを使います。

from ivory.core.variable import Variable  # isort:skip


def train(optimizer):
    x = Variable((2,))  # 2次元ベクトルのパラメータ
    x.data = np.array([-7.0, 2.0])
    optimizer.set_variables([x])

    for i in range(30):
        yield x.data.copy()  # inplaceで値が更新されるため、コピー値を返す。
        x.grad = df(x.data)
        optimizer.update()  # ここで、x.gradに従って、x.dataが更新される。


# {{ import matplotlib.pyplot as plt }}訓練と可視化を行います。
import matplotlib.pyplot as plt  # isort:skip

from ivory.core.optimizer import SGD, AdaGrad, Adam, Momentum  # isort:skip

optimizers = [
    SGD(learning_rate=0.95),
    Momentum(learning_rate=0.1),
    AdaGrad(learning_rate=1.5),
    Adam(learning_rate=0.3),
]
outs = [np.array(list(train(optimizer))) for optimizer in optimizers]

x = np.arange(-8, 8, 0.01)
y = np.arange(-3, 3, 0.01)
xs, ys = np.meshgrid(x, y)
zs = f([xs, ys])
zs[zs > 7] = 0

for k, out in enumerate(outs):
    plt.subplot(2, 2, k + 1)
    plt.contour(xs, ys, zs)
    plt.plot(out[:, 0], out[:, 1], "o-", color="red")
    plt.plot(0, 0, "+")
    name = optimizers[k].name
    plt.title(name)

plt.tight_layout()
plt.show()

# # 6.1.8 MNISTデータセットによる更新手法の比較

# `ch06/optimizer_compare_mnist.py`および`common/multi_layer_net.py`を参考にして、上述の
# 更新手法を比較してみます。

# データセットを準備します。

from ivory.datasets.mnist import load_dataset  # isort:skip

data = load_dataset(train_only=True)  # 今回は訓練データのみ使う
data.batch_size = 128
data.epochs = -1
data.random = True
data

# ネットワークを構築します。
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 784),
    (4, "affine", 100, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
model = sequential(net)
model.layers


# 訓練を行うジェネレータ関数を定義します。
def train(optimizer):  # type: ignore
    print(optimizer.name)
    for variable in model.weight_variables:  # 重みを初期化
        variable.data = variable.init()
    optimizer.set_model(model)

    for _, (x, t) in zip(range(2000), data):
        model.set_data(x, t)
        model.forward()
        yield model.loss
        model.backward()
        optimizer.update()


# 訓練を実行します。
optimizers = [SGD(), Momentum(), AdaGrad(), Adam()]
result = [list(train(optimizer)) for optimizer in optimizers]

# 可視化を行います。
from scipy.signal import savgol_filter  # isort:skip

markers = ["o", "x", "s", "D"]
for k, x in enumerate(savgol_filter(result, 9, 3)):
    plt.plot(x, marker=markers[k], markevery=100, label=optimizers[k].name)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.xlim(0, 2000)
plt.ylim(0, 1)
plt.legend()
plt.show()
