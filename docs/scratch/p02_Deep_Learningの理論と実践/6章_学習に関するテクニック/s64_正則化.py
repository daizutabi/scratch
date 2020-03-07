# ## 正則化

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 6章4節
# の内容を学習しながら、Ivoryライブラリで正則化の評価を行います。

# ### 過学習
# 過学習を起こしてみます。

from ivory.core.trainer import sequential
from ivory.datasets.mnist import load_dataset

data_train, data_test = load_dataset()
data_train.length = 300  # データセットの大きさを制限する。
data_test.length = 300
data_train.batch_size = 100
data_train.epochs = 200
data_train.random = True
epoch_data = {"train": data_train[:], "test": data_test[:]}

net = [
    ("input", 784),
    (6, "affine", 100, "relu"),  # 深いネットワークを作成する。
    ("affine", 10, "softmax_cross_entropy"),
]

trainer = sequential(net, metrics=["accuracy"])
trainer = trainer.fit(data_train, epoch_data=epoch_data)
df = trainer.to_frame()
df.tail()


# -
import altair as alt  # isort:skip


def plot(df):
    y = alt.Y("accuracy", scale=alt.Scale(domain=[0, 1]))
    return (
        alt.Chart(df)
        .mark_line()
        .encode(x="epoch", y=y, color="data")
        .properties(width=200, height=160)
    )


plot(df)
# # 6.4.2 Weight decay

# {{ from ivory.utils.latex import Matrix }} {{ W = Matrix('w') }}重みを${{W}}$としたと
# きに、損失関数に正則化項$\frac12\lambda{{W}}^2$を加えて、大きな重みに対してペナルティを
# 課します。また、逆伝搬においては、正則化項の微分$\lambda{{W}}$を加算します。Affineレイ
# ヤは、`weight_decay`状態を持っています。`Trainer`インスタンスからAffineレイヤをひとつ取
# 得します。
affine = trainer.model.layers[0]
affine.states
# `weight_decay`属性で直接アクセスできます。
affine.weight_decay  # type:ignore
# 値を確認します。
affine.weight_decay.d  # type:ignore
# 初期値はゼロです。つまり、これまでWeight Decayを使ってきませんでした。有効化するために
# ゼロ以上の値に設定したいと思います。ただし、現在のネットワークには複数のAffineレイヤが
# あります。一括で設定するために、`Trainer`インスタンスの`init`メソッドをここでも用います
# 。`Trainer`インスタンスはその`init`メソッドで、ネットワークの初期化を行います。こ
# のとき、個々の変数がどのように初期化されるかまでは把握しません。レイヤが`init`メソッドに
# 与えられたキーワード引数のキーを属性に持っていた場合には、その値で初期化します。

df = trainer.init(weight_decay=0.1).to_frame()
df.tail()

# -
plot(df)


# ### Dropout

# 活性化レイヤの後にDropout層を追加します。

net = [
    ("input", 784),
    (6, "affine", 100, "relu", "dropout"),
    ("affine", 10, "softmax_cross_entropy"),
]

# `Trainer`インスタンスは、同じ設定でネットワークだけ別のものに入れ替えることができます
# 。

trainer.set_net(net)
layers = trainer.model.layers
layers[:4]

# Dropoutレイヤは`dropout_ratio`状態を持っています。
layers[2].dropout_ratio  # type:ignore
# 初期値を確認します。
layers[2].dropout_ratio.d  # type:ignore
# 訓練を実施します。値を0.15に設定して訓練します。
data_train.epochs = 300
df = trainer.init(dropout_ratio=0.15).to_frame()
plot(df)
