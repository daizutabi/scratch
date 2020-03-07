# # 3.4 CBOWモデルの実装

# ### 学習コードの実装

# カスタムデータセットを準備します。
from ivory.common.dataset import Dataset
from ivory.common.util import (convert_one_hot, create_contexts_target,
                               preprocess)

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
vocab_size = len(word_to_id)
contexts = convert_one_hot(contexts, vocab_size)
data = Dataset((contexts, target))
data.batch_size = 3
data.epochs = 1000
data.shuffle()
data

# Ivoryライブラリでは重みを共有して複数入力をとるMatMulMeanレイヤを実装しています。これ
# を使って、CBOWモデルを組み立てます。
from ivory.core.trainer import sequential  # isort:skip

net = [("input", 2, 7), ("matmulmean", 5), ("matmul", 7, "softmax_cross_entropy")]
trainer = sequential(net, optimizer="adam").init(std=0.01)
trainer.model.layers
# 訓練を実行します。
df = trainer.fit(data, epoch_data=data[:]).to_frame()
df.tail()
# 可視化します。
import altair as alt  # type:ignore

alt.Chart(df).mark_line().encode(x='epoch', y='loss').properties(width=200, height=160)
