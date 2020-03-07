# # 6.6 まとめ

# 前節でさらなる改善を施したRNNの訓練は、独立した[スクリプ
# ト](https://github.com/daizutabi/ivory/blob/master/docs/script/train_better_rnnlm.py)で
# 行いました。結果を検証します。
import os

import altair as alt
import pandas as pd

import ivory

directory = os.path.dirname(ivory.__file__)
directory = os.path.join(directory, "../docs/script")
path = os.path.join(directory, "better_rnnlm_ppl.csv")
df = pd.read_csv(path)
df.tail()
# プロットします。
alt.Chart(data=df).mark_line().encode(x="epoch", y="ppl_val").properties(
    width=200, height=160
)

# テストデータでのパープレキシティを求めてみます。
# 高速化のためにGPUを使います。
from ivory.common.context import np  # isort:skip
np.context = 'gpu'

# PTBデータセットを読み出します。
from ivory.common.dataset import TimeDataset  # isort:skip
from ivory.utils.repository import import_module  # isort:skip

ptb = import_module("scratch2/dataset/ptb")
corpus_test, _, _ = ptb.load_data("test")
x, t = corpus_test[:-1], corpus_test[1:]
data = TimeDataset((x, t), time_size=35, batch_size=20)
data

# モデルを作成します。
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 10000),
    ("embedding", 650),
    ("lstm", 650),
    ("lstm", 650),
    ("affine", 10000, "softmax_cross_entropy"),
]
model = sequential(net)

# 重みの共有をします。
em = model.layers[0]
affine = model.layers[-2]
affine.W.share_variable(em.W, transpose=True)  # type:ignore
model.build()

# 学習済みの重みを読み出します。
import pickle  # isort:skip

with open(os.path.join(directory, 'better_rnnlm.pkl'), 'rb') as f:
    weights = pickle.load(f)

for v, weight in zip(model.weight_variables, weights):
    v.data = np.asarray(weight)

# テストデータでのパープレキシティを求めます。
count = 0
total_loss = 0.0
for x, t in data:
    model.set_data(x, t)
    model.forward()
    total_loss += model.loss
    count += 1
print(np.exp(total_loss / count))

# 「ゼロから作るDeep Learning ❷」と同等の結果が得られました。
