# ## BatchNormalization

# ### Batch Normalizationのアルゴリズム

# -hide
import sympy

sympy.symbols("x")

# ミニバッチを単位として、データの分布が平均ゼロ、分散１になるように正規化します。

# $$\mu_B \leftarrow \frac1m\sum_{i=1}^mx_i$$

# $$\sigma_B^2 \leftarrow \frac1m\sum_{i=1}^m(x_i-\mu_B)^2$$

# $$\hat{x}_i \leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$

# $$y_i \leftarrow \gamma\hat{x}_i+\beta$$

# 逆伝搬で求めたいのは、

# $$\frac{\partial L}{\partial x_i}, \frac{\partial L}{\partial \gamma}, \frac{\partial
# L}{\partial \beta}$$

# $$\frac{\partial L}{\partial \beta} = \sum_i\frac{\partial L}{\partial
# y_i}\frac{\partial y_i}{\partial \beta} = \sum_i\frac{\partial L}{\partial y_i}$$

# $$\frac{\partial L}{\partial \gamma} = \sum_i\frac{\partial L}{\partial
# y_i}\frac{\partial y_i}{\partial \gamma} = \sum_i\frac{\partial L}{\partial
# y_i}\hat{x}_i$$

# $$\frac{\partial L}{\partial \hat{x}_i} = \sum_j\frac{\partial L}{\partial
# y_j}\frac{\partial y_j}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i}\gamma$$

# $$x_i = \sigma_B\hat{x}_i + \mu_B$$

# です
# 。{{path="understanding-the-gradient-flow-through-the-batch-normalization-layer.html"}}逆
# 伝搬の導出は、Frederik Kratzertのブログ[「Understanding the backward pass through Batch
# Normalization Layer」](https://kratzert.github.io/2016/02/12/{{path}})に詳しい解説があ
# ります。

# Batch Normalizationクラスを実装します。「ゼロから作るDeep Learning」
# の`common/layers.py`を参考にしています。以下では実装の中心となる`forward_2d`メソッド
# と`backward_2d`メソッドを記載します。

# {{ from ivory.layers.normalization import BatchNormalization }}
# ##Code <code>BatchNormalization.forward_2d</code>および<code>backward_2d</code>
# {{ [BatchNormalization.forward_2d, BatchNormalization.backward_2d] # inspect }}

# さて、上記のクラスで正しいBatch Normalizationを実現できているか、確認します。


import numpy as np  # isort:skip
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 20),
    (2, "affine", 4, "batch_normalization", "relu"),
    ("affine", 5, "softmax_cross_entropy"),
]

model = sequential(net)
model.layers

# BatchNormalizationレイヤのパラメータを見てみます。
bn = model.layers[1]
bn.parameters
# 正規化後の変換を表す$\gamma$と$\beta$があります。また、状態変数を持ちます。これらは、テ
# スト時に用いる移動平均`running_mean`と移動分散`running_var`、および、訓練状態か否かのフ
# ラッグ`train`です。
print("running mean:", bn.running_mean.d)  # type:ignore
print("running var: ", bn.running_var.d)  # type:ignore
print("train:       ", bn.train.d)  # type:ignore

# 次に、BatchNormalizationレイヤの前後でのデータの変化をみます。
xv, tv = model.data_input_variables
layers = model.layers
affine1, norm1 = layers[0].y, layers[1].y
affine2, norm2 = layers[3].y, layers[4].y
# -
batch_size = 100
x = np.random.randn(batch_size, *xv.shape)
high = layers[-1].x.shape[0]
t = np.random.randint(0, high, (batch_size, *tv.shape))
model.set_data(x, t)

model.forward()
model.backward()
# -
print(affine1.d.mean(axis=0))
print(affine1.d.std(axis=0))
print(affine2.d.mean(axis=0))
print(affine2.d.std(axis=0))
print(norm1.d.mean(axis=0))
print(norm1.d.std(axis=0))
print(norm2.d.mean(axis=0))
print(norm2.d.std(axis=0))
# このように、Batch Normalizationの出力は、ユニットごとのバッチ内分布が平均0、標準偏差1に
# 正規化されています。次に勾配確認を行います。

for v in model.grad_variables:
    print(v.parameters[0].name, model.gradient_error(v))


# 差分が小さい値になっていることが分かります。以上で、Batch Normalizationレイヤが実装でき
# ました。
