# # 6.5 RNNLMのさらなる改善

# 高速化のためにGPUを使います。
from ivory.common.context import np
np.context = 'gpu'

# PTBデータセットを読み出します。
from ivory.common.dataset import TimeDataset  # isort:skip
from ivory.utils.repository import import_module  # isort:skip

ptb = import_module("scratch2/dataset/ptb")
corpus, _, _ = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
corpus_test, _, _ = ptb.load_data("test")
vocab_size = int(max(corpus) + 1)
x, t = corpus[:-1], corpus[1:]
data = TimeDataset((x, t), time_size=35, batch_size=20)
data.epochs = 1
data

# ハイパーパラメータの設定を行います。
wordvec_size = 650
hidden_size = 650
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# モデルを作成します。
from ivory.core.trainer import sequential  # isort:skip

net = [
    ("input", vocab_size),
    ("embedding", wordvec_size, "dropout"),
    ("lstm", hidden_size, "dropout"),
    ("lstm", hidden_size, "dropout"),
    ("affine", vocab_size, "softmax_cross_entropy"),
]
trainer = sequential(net, optimizer="sgd", metrics=["loss"])
trainer.optimizer.learning_rate = lr
trainer.max_grad = max_grad
model = trainer.model
for layer in model.layers:
    print(layer)
# 重みの初期値とドロップアウト率を設定します。
from ivory.common.context import np  # isort:skip

model.init(std="xavier", dropout_ratio=dropout)
for p in model.weights:
    if p.name != "b":
        std1 = f"{float(p.d.std()):.03f}"
        std2 = f"{float(np.sqrt(1/p.d.shape[0])):.03f}"
        print(p.layer.name, p.name, std1, std2, type(p.d), p.d.dtype)

for layer in model.layers:
    if layer.name.startswith('Dropout'):
        print(layer.name, layer.dropout_ratio.d)  # type:ignore
# 重みの共有をします。
em = model.layers[0]
affine = model.layers[-2]
affine.W.share_variable(em.W, transpose=True)  # type:ignore
trainer.build()
for v in trainer.optimizer.variables:
    print(v)

# 訓練を実施します。
trainer.fit(data)
it = iter(trainer)
loss = next(it)[1]
print(data.iteration, int(np.exp(loss)))

for i in range(8):
    loss = 0.0
    for _ in range(20):
        loss += next(it)[1]
    loss /= 20.0
    print(data.iteration, int(np.exp(loss)))

# 「ゼロから作るDeep Learning ❷」の`ch06/train_better_rnnlm.py`の実行結果の冒頭を記載しま
# す。

# ~~~bash
# | epoch 1 |  iter 1 / 1327 | time 2[s] | perplexity 9999.86
# | epoch 1 |  iter 21 / 1327 | time 60[s] | perplexity 4233.17
# | epoch 1 |  iter 41 / 1327 | time 116[s] | perplexity 1645.35
# | epoch 1 |  iter 61 / 1327 | time 172[s] | perplexity 1346.09
# | epoch 1 |  iter 81 / 1327 | time 227[s] | perplexity 1022.61
# | epoch 1 |  iter 101 / 1327 | time 283[s] | perplexity 845.07
# | epoch 1 |  iter 121 / 1327 | time 339[s] | perplexity 810.82
# | epoch 1 |  iter 141 / 1327 | time 395[s] | perplexity 749.34
# | epoch 1 |  iter 161 / 1327 | time 451[s] | perplexity 685.36
# ~~~

# 実際の訓練は独立したスクリプトファイルを作成して実行します。次節で、結果を検証します。
