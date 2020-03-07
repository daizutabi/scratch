# ## トレーナー

# 「ゼロから作るDeep Learning」では、学習の効率化のために`Trainer`クラスを導入しています
# 。Ivoryライブラリでも`Trainer`クラスを実装し、少ないコードで学習ができるようにしていま
# す。

# データセットを用意します。
from ivory.datasets.mnist import load_dataset  # isort:skip

data = load_dataset(train_only=True)
data.length = 1000  # データセットの大きさを制限します。
data.batch_size = 100
data.epochs = 20
data.random = True
data

# `Trainer`クラスはレイヤ表現を引数にとる`ivory.common.trainer`モジュール
# の`sequential`関数で作成できます。
from ivory.core.trainer import sequential  # isort:skip

net = [("input", 784), ("affine", 50, "relu"), ("affine", 10, "softmax_cross_entropy")]
trainer = sequential(net)
print(trainer.model)
print(trainer.optimizer)

# `Trainer`インスタンスのレイヤパラメータを初期化するには、`init`メソッドを呼び出します
# 。オプショナルの`std`キーワード引数を指定すると、標準偏差を設定できます。
from ivory.common.context import np  # isort:skip

W = trainer.model.weight_variables[0]
print(W.data.std(), np.sqrt(2 / W.shape[0]))  # type:ignore
trainer.init(std=100)
print(W.data.std())  # type:ignore
trainer.init(std="he")
print(W.data.std())  # type:ignore

# レイヤの状態変数は、名前を引数に渡すことで値を設定できます。キーワードを指定しなければ
# 、初期値に戻ります。
wd = trainer.model.state_variables[0]
print(wd.data)
trainer.init(weight_decay=1)
print(wd.data)
trainer.init()
print(wd.data)

# 実際に学習してみます。`init`メソッドは自分自身を返すので、呼び出しをチェインできます
# 。`fit`メソッドも同様に訓練データの設定をした後、自分自身を返します。

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
