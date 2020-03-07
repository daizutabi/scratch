# # 7.5 CNNの実装

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 7章5節
# のCNNの実装を、Ivoryライブラリで再現します。

# データセットを用意します。これまでと違い、画像を平坦化しません。
from ivory.datasets.mnist import load_dataset  # isort:skip

data_train, data_test = load_dataset(flatten=False)
print(data_train)
print(data_train.shape)

# 「ゼロから作るDeep Learning」の`SimpleConvNet`を作成します。学習のために、Trainerインス
# タンスを用意します。
from ivory.core.trainer import sequential  # isort:skip

net = [
    ("input", 1, 28, 28),
    ("convolution", 30, 5, 5, "relu"),
    ("pooling", 2, 2, "flatten"),
    ("affine", 100, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
trainer = sequential(net, optimizer="adam", metrics=["accuracy"])

# 学習を行います。
data_train.epochs = 20
data_train.batch_size = 100
data_train.shuffle()
data_test.shuffle()

# エポックごとの評価にスライス表記で取得したデータを使います。スライス表記にはバッチ数分
# を含むので、バッチサイズが100の訓練データに対しては、`data_train[:10]`で1000個分のデー
# タが取得できます。

epoch_data = {"train": data_train[:10], "test": data_test[:1000]}
len(epoch_data["train"][0]), len(epoch_data["test"][0])

# -
trainer.init(std=0.01)
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
