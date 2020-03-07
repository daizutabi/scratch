# ## ハイパーパラメータの検証

# # 6.5.3 ハイパーパラメータ最適化の実装
# `ch06/hyperparameter_optimization.py`を参考にしながら、ハイパーパラメータの最適化を行
# ってみます。

# 課題を定義します。
from ivory.core.trainer import sequential
from ivory.datasets.mnist import load_dataset

data = load_dataset(train_only=True)
data.length = 500
data.shuffle()
data.split((8, 2))
data.batch_size = 100
data.epochs = 50
epoch_data = {"train": data[0, :], "val": data[1, :]}

net = [
    ("input", 784),
    (6, "affine", 100, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]

trainer = sequential(net, metrics=["accuracy"])
trainer = trainer.fit(data, epoch_data=epoch_data)
trainer

# ハイパーパラメータのランダム探索を行うジェネレータを定義します。以下ではハイパーパラメ
# ータの探索範囲を「ゼロから作るDeep Learning」から少し変更しています。
import numpy as np  # isort:skip


def random_search():
    while True:
        weight_decay = 10 ** np.random.uniform(-10, -2)
        learning_rate = 10 ** np.random.uniform(-4, -1)
        trainer.optimizer.learning_rate = learning_rate
        trainer.init(weight_decay=weight_decay)
        df = trainer.to_frame()
        columns = list(df.columns)
        df["wd"] = weight_decay
        df["lr"] = learning_rate
        yield df[["wd", "lr"] + columns]


# 試してみます。
searcher = random_search()
print(next(searcher).tail(4))
print(next(searcher).tail(4))
print(next(searcher).tail(4))

# 100回のトライアルを行います。
import pandas as pd  # isort:skip

df = pd.concat([df for _, df in zip(range(100), searcher)])
len(df)

# ベスト5を可視化してみます。
acc = df.query("data == 'val'").groupby(["wd", "lr"])["accuracy"].max()
best = acc.sort_values(ascending=False).to_frame().reset_index()
best.index.name = "best"
best = best.reset_index()
best["best"] += 1
best[:5]

# -
import altair as alt  # isort:skip
df_best = pd.merge(df, best[["best", "wd", "lr"]][:5])
alt.Chart(df_best).mark_line().encode(
    x="epoch", y="accuracy", color="data", column="best"
).properties(width=100, height=120)
# -
x = alt.X("wd", scale=alt.Scale(type="log"))
y = alt.Y("lr", scale=alt.Scale(type="log"))
alt.Chart(best).mark_point().encode(
    x=x, y=y, color="accuracy", size="accuracy"
).properties(width=200, height=200)
