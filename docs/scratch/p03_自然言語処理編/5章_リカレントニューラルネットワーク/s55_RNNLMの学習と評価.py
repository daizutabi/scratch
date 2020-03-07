# # 5.7 RNNLMの学習と評価

# PTBデータセットを読み出します。
from ivory.common.dataset import TimeDataset
from ivory.utils.repository import import_module

ptb = import_module("scratch2/dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus = corpus[:1000]
vocab_size = int(max(corpus) + 1)
x, t = corpus[:-1], corpus[1:]
data = TimeDataset((x, t), time_size=5, batch_size=10)
data.epochs = 100
data

# ハイパーパラメータの設定を行います。
wordvec_size = 100
hidden_size = 100

# モデルを作成します。
from ivory.core.trainer import sequential  # isort:skip

net = [
    ("input", vocab_size),
    ("embedding", wordvec_size),
    ("rnn", hidden_size),
    ("affine", vocab_size, "softmax_cross_entropy"),
]
trainer = sequential(net, optimizer="sgd", metrics=["loss"])
trainer.optimizer.learning_rate = 0.1
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
# モデルに代入し、パープレキシティを計算してみます。
trainer.set_data(*data[0])
model.forward()
print(model.perplexity)

# 訓練を実施します。
trainer.fit(data)
df = trainer.to_frame()
df["epoch"] = df.iteration // len(data)
df = df.groupby("epoch").mean().reset_index()
df["ppl"] = np.exp(df.loss)
df.tail()

# 可視化します
import altair as alt  # isort:skip

alt.Chart(df).mark_line().encode(x="epoch", y="ppl").properties(width=300, height=200)
