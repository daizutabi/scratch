# # 4.3 改良版word2vecの実装

# モデルを構築する関数を用意します。
from ivory.common.context import np  # isort:skip
from ivory.core.model import Model, branch  # isort:skip


def cbow(vocab_size, window_size=5, hidden_size=100, sample_size=5, batch_size=100):
    em = [("input", 2 * window_size, vocab_size), ("embeddingmean", hidden_size)]
    ns = [("embeddingdot", vocab_size), "sigmoid_cross_entropy"]
    h = branch(em)[-1]
    losses = [branch(ns, h)[-1].loss for _ in range(sample_size + 1)]  # type:ignore
    model = Model(losses)
    # EmbeddingDotレイヤはひとつの重みを共有します。
    v = model.weights[1].variable
    for p in model.weights[2:]:
        p.set_variable(v)
    len(v.parameters)  # type:ignore
    # 正例と負例の正解ラベルを代入します。今後更新することがないので、`frozen`を`True`に設定します。
    v = model.data_input_variables[2]
    v.data = np.ones(batch_size, dtype=np.int32)
    v.frozen = True
    for v in model.data_input_variables[4::2]:
        v.data = np.zeros(batch_size, dtype=np.int32)
        v.frozen = True
    # 再度モデルをビルドします。
    model.build()
    # 重みを初期化します。
    model.init(std=0.01)
    return model


# まずは実験用のコーパスを準備します。
from ivory.common.dataset import ContextDataset  # isort:skip

corpus = [0, 1, 2, 3, 4, 1, 2, 3, 2]
data = ContextDataset(corpus, replace=True)

# ハイパーパラメータの設定を行います。
data.set_window_size(2)
data.negative_sample_size = 2
data.batch_size = 2
hidden_size = 10

# モデルを作成します。
model = cbow(
    data.vocab_size,
    window_size=data.window_size,
    hidden_size=hidden_size,
    sample_size=data.negative_sample_size,
    batch_size=data.batch_size,
)

# モデルの確認を行います。
model.data_input_variables
# -
model.frozen_input_variables
# -
model.weight_variables
# 重みの標準偏差を確認します。
for v in model.weight_variables:
    print(v.data.std())  # type:ignore

# 勾配確認のために、重みのビット精度を64ビットにします。
for v in model.weight_variables:
    v.data = v.data.astype(np.float64)

# モデルに代入してみます。
model.set_data(*data[0])
model.forward()
model.backward()
model.loss
# 数値微分による勾配確認を行います。


for v in model.grad_variables:
    print(model.gradient_error(v))
    print(v.grad[:2, :4])
    print(model.numerical_gradient(v)[:2, :4])
# 正しい結果が得られています。


# 次に、PTBデータセットを読み出します。
from ivory.utils.repository import import_module  # isort:skip

ptb = import_module("scratch2/dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data("train")
data = ContextDataset(corpus, window_size=5, replace=True)

# ハイパーパラメータの設定を行います。
data.negative_sample_size = 5
data.batch_size = 100
hidden_size = 100

# モデルを作成します。
model = cbow(
    data.vocab_size,
    window_size=data.window_size,
    hidden_size=hidden_size,
    sample_size=data.negative_sample_size,
    batch_size=data.batch_size,
)

# モデルに代入してみます。
model.set_data(*data[0])
model.forward()
model.backward()
model.loss

# トレーナーに登録します。
from ivory.core.trainer import Trainer  # isort:skip
from ivory.core.optimizer import Adam  # isort:skip

trainer = Trainer(model, optimizer=Adam(), dataset=data, metrics=['loss'])
trainer.init(std=0.01)
# 訓練を実施します。
for _ in zip(range(201), trainer):
    if data.iteration % 20 == 0:
        print(data.iteration, model.loss)


# GPUを使ってみます。
np.context = 'gpu'
data = ContextDataset(corpus, window_size=5, replace=True)
data.negative_sample_size = 5
data.batch_size = 100

model = cbow(
    data.vocab_size,
    window_size=data.window_size,
    hidden_size=hidden_size,
    sample_size=data.negative_sample_size,
    batch_size=data.batch_size,
)

trainer = Trainer(model, optimizer=Adam(), dataset=data, metrics=['loss'])
trainer.init(std=0.01)
# 訓練を実施します。
for _ in zip(range(201), trainer):
    if data.iteration % 20 == 0:
        print(data.iteration, model.loss)
