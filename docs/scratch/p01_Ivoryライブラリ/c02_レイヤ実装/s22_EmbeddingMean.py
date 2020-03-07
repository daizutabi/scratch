# ## EmbeddingMean

# Embeddingレイヤの入力は、バッチ数分のラベル値でした。よって、EmbeddingMeanレイヤへは
# 、(バッチ数, 入力数)のラベルとなります。

# Embeddingレイヤの`forward`メソッドは以下の通りでした。

# ~~~python
#    def forward(self):
#        self.y.d = self.W.d[self.x.d]
# ~~~

# まずはこれを拡張します。形状を（バッチ数、入力数、入力の次元、出力の次元）とします。
import numpy as np

N, I, L, M = 2, 3, 5, 3
# 重みは、
w = np.random.randn(L, M)
print(w)
# 入力は、
x = np.random.randint(0, L, (N, I))
print(x)
# 各入力に対する出力は、
print(w[x.T])
# となり、複数の入力に対する出力を平均化するので、
print(w[x.T].sum(axis=0) / I)
# となりますが、同じことは、
print(w[x].sum(axis=1) / I)
# でもできます。以上が順伝搬になります。ここまでを実装します。

from ivory.core.layer import Layer  # isort:skip


class EmbeddingMean(Layer):
    def init(self):
        self.W = self.add_weight(self.shape[1:]).randn()

    def forward(self):
        self.y.d = self.W.d[self.x.d].sum(axis=1) / self.shape[0]


# すでにあるMatMulMeanレイヤと比較します。
from ivory.core.model import sequential  # isort:skip
from ivory.common.util import convert_one_hot  # isort:skip

N, I, L, M = 20, 10, 4, 3
net_mat = [("input", I, L), ("matmulmean", M, "softmax_cross_entropy")]
model_mat = sequential(net_mat)
net_em = [("input", I, L), ("embeddingmean", M, "softmax_cross_entropy")]
model_em = sequential(net_em)

# ランダムな入力を作成します。
x = np.random.randint(0, L, (N, I))
t = np.random.randint(0, M, N)
# MatMulレイヤへ入力するために、one-hot表現に変換します。
model_mat.set_data(convert_one_hot(x, L), t)
# Embeddingレイヤへは、そのまま入力します。
model_em.set_data(x, t)
# 両者を比較するために重みを同じ値に設定します。
mat, em = model_mat.layers[0], model_em.layers[0]
em.share_weight_variables(mat)
# 順伝搬を比較します。
model_mat.forward()
model_em.forward()
model_mat.loss, model_em.loss


# 次に逆伝搬を実装します。ここでも勾配確認のテクニックを使います。
# 数値微分による勾配を求めます。
model_em.numerical_gradient(em.W.variable)  # type:ignore

# EmbeddingMeanレイヤの出力の勾配を求めます。
try:
    model_em.backward()
except AttributeError:  # まだ、backwardメソッドを定義していないため
    pass

# Embeddingレイヤの逆伝搬と同じことをしますが、形状を合わせるために転置します。
grad = np.zeros_like(em.W.d)  # type:ignore
np.add.at(grad, em.x.d.T, em.y.g)
grad /= em.shape[0]
grad

# 一致が確認できました。最終的な実装は以下のようになります。

# {{ from ivory.layers.embedding import EmbeddingMean }}
# ##Code <code>EmbeddingMean</code>クラス
# {{ EmbeddingMean # inspect }}

# MatMulMeanクラスの代わりに使ってみます。
N, I, L, M = 20, 10, 4, 3
net_mat = [("input", I, L), ("matmulmean", M, "softmax_cross_entropy")]
model_mat = sequential(net_mat)
net_em = [("input", I, L), ("embeddingmean", M, "softmax_cross_entropy")]
model_em = sequential(net_em)

# ランダムな入力を作成します。
x = np.random.randint(0, L, (N, I))
t = np.random.randint(0, M, N)
# MatMulレイヤへ入力するために、one-hot表現に変換します。
model_mat.set_data(convert_one_hot(x, L), t)
# Embeddingレイヤへは、そのまま入力します。
model_em.set_data(x, t)
# 両者を比較するために重みを同じ値に設定します。
mat, em = model_mat.layers[0], model_em.layers[0]
em.share_weight_variables(mat)
# 順伝搬を比較します。
model_mat.forward()
model_em.forward()
model_mat.loss, model_em.loss

# 逆伝搬を比較します。
model_mat.backward()
model_em.backward()
mat.W.g, em.W.g  # type:ignore

# 一致が確認できました。
