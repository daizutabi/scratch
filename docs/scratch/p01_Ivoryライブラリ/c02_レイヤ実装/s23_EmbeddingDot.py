# ## EmbeddingDot

# 「ゼロから作るDeep Learning ❷」に従ったEmbeddingDotレイヤの実装は以下のようになります。

# {{ from ivory.layers.embedding import EmbeddingDot }}
# ##Code <code>EmbeddingDot</code>クラス
# {{ EmbeddingDot # inspect }}

# 動作を確かめます。形状（バッチ数、単語ベクトルの次元、ターゲットラベルの次元）を以下の
# 通りとします。
N, L, M = 10, 4, 5

from ivory.layers.embedding import EmbeddingDot

dot = EmbeddingDot((L, M))
print(dot)
print(dot.x)
print(dot.t)
print(dot.W)
print(dot.y)

# これまでのレイヤと違い、レイヤの形状と重みの形状が逆転していることに注意します。また、
# 出力は内積の結果なので、損失関数と同様にスカラー値となります。2つ目の入
# 力`t`がEmbeddingレイヤへの入力でこちらもラベル表現なのでスカラー値となります。（バッチ
# 数は除きます。）

# ランダムな入力他を作成します。
import numpy as np  # isort:skip

x = np.random.randn(N, L)
t = np.random.randint(0, M, N)
w = np.random.randn(M, L)
# レイヤに入力し、順伝搬します。
dot.set_variables()
dot.x.variable.data = x
dot.t.variable.data = t
dot.W.variable.data = w
dot.forward()
dot.y.d[:3]
# バッチ数分の内積値が出力されました。上記が正しいか、確かめます。
[xi @ w[ti] for xi, ti in zip(x, t)][:3]

# 逆伝搬を検証するために、数値微分による勾配確認を行います。これまでと違い
# 、SigmoidCrossEntropyレイヤを用います。
from ivory.core.model import sequential  # isort:skip

net = [("input", L), ('embeddingdot', M), ('sigmoid_cross_entropy')]
model = sequential(net)
model.data_input_variables

# 正解ラベルを乱数で生成します。
t2 = np.random.randint(0, 2, N)
# データを入力します。
model.set_data(x, t, t2)
# 数値微分による勾配を求めます。
W = model.layers[0].W  # type:ignore
model.numerical_gradient(W.variable)
# 逆伝搬による勾配を求めます。
model.forward()
model.backward()
W.g
# 一致した結果が得られました。

# ここで`backward`メソッドを再掲します。

# ##Code <code>EmbeddingDot.backward</code>メソッド
# {{ EmbeddingDot.backward # inspect }}

# Affineレイヤと同じように出力の勾配に内積の相手側を掛けた行列を入力側に逆伝搬しています
# 。ただし、内積なので、要素ごとの積になっています。
