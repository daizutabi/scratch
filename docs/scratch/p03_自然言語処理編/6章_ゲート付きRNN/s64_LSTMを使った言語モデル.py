# # 6.4 LSTMを使った言語モデル

# PTBデータセットを読み出します。
from ivory.common.dataset import TimeDataset
from ivory.utils.repository import import_module

ptb = import_module("scratch2/dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = int(max(corpus) + 1)
x, t = corpus[:-1], corpus[1:]
data = TimeDataset((x, t), time_size=35, batch_size=20)
data.epochs = 4
data

# ハイパーパラメータの設定を行います。
wordvec_size = 100
hidden_size = 100
lr = 20
max_grad = 0.25

# モデルを作成します。
from ivory.core.trainer import sequential  # isort:skip

net = [
    ("input", vocab_size),
    ("embedding", wordvec_size),
    ("lstm", hidden_size),
    ("affine", vocab_size, "softmax_cross_entropy"),
]
trainer = sequential(net, optimizer="sgd", metrics=["loss"])
trainer.optimizer.learning_rate = lr
trainer.max_grad = max_grad
model = trainer.model
for layer in model.layers:
    print(layer)
# 重みの初期値を設定します。
from ivory.common.context import np  # isort:skip

model.init(std="xavier")
for p in model.weights:
    if p.name != "b":
        std1, std2 = f"{p.d.std():.03f}", f"{np.sqrt(1/p.d.shape[0]):.03f}"
        print(p.layer.name, p.name, std1, std2)

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
    print(data.iteration + 1, int(np.exp(loss)))

# 「ゼロから作るDeep Learning ❷」の「図6-27 ターミナルの出力結果」と同等の結果が得られま
# した。参考までに、`ch06/train_rnnlm.py`の実行結果の冒頭を記載します。

# ~~~bash
# | epoch 1 |  iter 1 / 1327 | time 0[s] | perplexity 10000.64
# | epoch 1 |  iter 21 / 1327 | time 6[s] | perplexity 3112.58
# | epoch 1 |  iter 41 / 1327 | time 12[s] | perplexity 1260.55
# | epoch 1 |  iter 61 / 1327 | time 18[s] | perplexity 966.21
# | epoch 1 |  iter 81 / 1327 | time 24[s] | perplexity 815.49
# | epoch 1 |  iter 101 / 1327 | time 30[s] | perplexity 681.86
# | epoch 1 |  iter 121 / 1327 | time 37[s] | perplexity 645.88
# | epoch 1 |  iter 141 / 1327 | time 44[s] | perplexity 608.60
# | epoch 1 |  iter 161 / 1327 | time 50[s] | perplexity 592.78
# ~~~
