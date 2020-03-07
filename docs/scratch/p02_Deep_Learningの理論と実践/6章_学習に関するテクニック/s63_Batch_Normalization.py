# ## Batch Normalization

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 6章3節
# の内容を学習しながら、IvoryライブラリでBatch Normalizationの評価を行います。「ゼロから
# 作るDeep Learning」では、次節から学習の効率化のために`Trainer`クラスを導入しますが、こ
# こでは一足先に活用していきます。

# ### Batch Normalizationの評価

# 「ゼロから作るDeep Learning」のサンプルコードを参考にしながらIvoryライブラリでの評価を
# 行います。「拡張版の全結合による多層ニューラルネットワーク」
# は`common/multi_layer_net_extend.py`の中の`MultiLayerNetExtend`クラスで定義されています
# 。また、実際の訓練は、`ch06/batch_norm_test.py`で実行されます。

# データセットを用意します。
from ivory.datasets.mnist import load_dataset  # isort:skip

data = load_dataset(train_only=True)
data.length = 1000  # データセットの大きさを制限します。
data.batch_size = 100
data.epochs = 20
data.random = True
data


# Batch Normalizationレイヤの有無を指定して、ネットワークを作り分ける関数を定義します。
def create_net(batch_norm):
    bn = ("batch_normalization",) if batch_norm else ()
    return [
        ("input", 784),
        (5, "affine", 100, *bn, "relu"),
        ("affine", 10, "softmax_cross_entropy"),
    ]


net = create_net(True)
net

# 次に導入する`Trainer`クラスは上述のようなレイヤ表現を引数にとる`ivory.common.trainer`モ
# ジュールの`sequential`関数で作成できます。ちょうど`Layer`インスタンスのリスト
# を`ivory.common.layer`モジュールの`sequential`関数で作成していたのと同じです。

from ivory.core.trainer import sequential  # isort:skip

trainer = sequential(net)
print(trainer.model.losses)
print(trainer.optimizer)
# `Trainer`インスタンスのレイヤパラメータを初期化するには、`init`メソッドを呼び出します
# 。オプショナルの`std`キーワード引数を指定すると、標準偏差を設定できます。
W = trainer.model.weights[0]
print(W.d.std())
trainer.init(std=100)
print(W.d.std())  # type:ignore
trainer.init(std="he")
print(W.d.std())  # type:ignore
# BatchNormalizationレイヤの`train`状態を見てみます。
bn = trainer.model.layers[1]
bn.train.d  # type:ignore
# `False`にした後初期化します。
bn.train.d = False  # type:ignore
trainer.init(std="he")
bn.train.d  # type:ignore
# `True`に戻っています。

# 実際に学習してみます。`init`メソッドは自分自身を返すので、呼び出しをチェインできます
# 。`fit`メソッドも同様に訓練データの設定をした後、自分自身を返します。
net = create_net(True)
trainer = sequential(net, metrics=["accuracy"]).init(std=0.1)
trainer = trainer.fit(data, epoch_data=data[:])
trainer
# 実際の訓練はイタレータを作って行います。
it = iter(trainer)
print(next(it))
print(next(it))
print(next(it))
# `to_frame`メソッドは訓練を行った後に結果をデータフレームで返します。
df = trainer.to_frame()
df.tail()
# 重みの初期値の標準偏差、および、Batch Normalizationレイヤの有無をモデルのハイパーパラメ
# ータとします。これらをマトリックスにして学習の実験を行います。これには
# 、`ivory.core.trainer`モジュールの`product`メソッドを使います。`product`メソッドは、
# 標準モジュール`itertools`の`product`関数のように動作します。第1引数は、第2引数以降を受
# け取って`Trainer`インスタンスを返す関数です。戻り値は`Product`クラスのインスタンスであ
# り、通常の`Trainer`インスタンスと同様に`fit`メソッドを持ちます。

import numpy as np  # isort:skip
from ivory.core.trainer import product  # isort:skip


def trainer(std, batch_norm):  # type:ignore
    net = create_net(batch_norm)
    return sequential(net, metrics=["accuracy"]).init(std=std)


stds = np.logspace(0, -4, num=16)
prod = product(trainer, stds, [True, False])  # type:ignore
prod = prod.fit(data, epoch_data=data[:])
prod
# -

it = iter(prod)
print(next(it))
print(next(it))
print(next(it))


# 訓練を実行し、結果をデータフレームで返します。
df = prod.to_frame(columns=["std", "bn"])
df.tail()
# 可視化します。
import altair as alt  # isort:skip


def plot(std, df):
    y = alt.Y("accuracy", scale=alt.Scale(domain=[0, 1]))
    return (
        alt.Chart(df)
        .mark_line()
        .encode(x="epoch", y=y, color="bn")
        .properties(width=80, height=80, title=f"std={std:.05f}")
    )


charts = [plot(*x) for x in df.groupby("std")][::-1]
alt.vconcat(*(alt.hconcat(*charts[k : k + 4]) for k in range(0, 16, 4)))
