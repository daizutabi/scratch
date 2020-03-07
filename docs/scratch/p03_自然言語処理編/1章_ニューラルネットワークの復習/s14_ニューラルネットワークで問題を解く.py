# # 1.4 ニューラルネットワークで問題を解く

# [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/)
# 1章4節のスパイラルデータの学習を再現します。

from ivory.utils.repository import import_module

spiral = import_module("scratch2/dataset/spiral")
x, t = spiral.load_data()
t = t.argmax(axis=1)
print(x.shape)
print(t.shape)

# カスタムのデータセットを作成します。
from ivory.common.dataset import Dataset  # isort:skip

data = Dataset((x, t), name="spiral")
data.shuffle()
data.epochs = 300
data.batch_size = 30
data

# Trainerインスタンスを作成します。
from ivory.core.trainer import sequential  # isort:skip

net = [("input", 2), ("affine", 10, "sigmoid"), ("affine", 3, "softmax_cross_entropy")]

trainer = sequential(net, metrics=["loss"])
trainer.optimizer.learning_rate = 1
trainer = trainer.fit(data, epoch_data=data[:])
df = trainer.to_frame()
df.tail()


# 損失の履歴をプロットします。
import altair as alt  # isort:skip

df_mean = df.rolling(10, min_periods=0).mean()
alt.Chart(df_mean).mark_line().encode(x="epoch", y="loss").properties(
    width=200, height=160
)


# 分類結果を可視化します。{{ import matplotlib.pyplot as plt }}
import matplotlib.pyplot as plt  # isort:skip
import numpy as np  # isort:skip
import pandas as pd  # isort:skip

x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
X = np.c_[x.ravel(), y.ravel()]
z = trainer.predict(X).reshape(x.shape)
plt.contourf(x, y, z)
plt.axis("off")

x, t = data[:]
df = pd.DataFrame(x, columns=["x", "y"])
df["t"] = t
markers = ["o", "x", "^"]
for (key, sub), marker in zip(df.groupby("t"), markers):
    plt.scatter(sub.x, sub.y, s=40, marker=marker)
plt.show()
