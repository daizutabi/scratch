# ## MatMul/Embeddingレイヤ

# ### 定式化

# -hide
from ivory.utils.latex import Matrix

X = Matrix("x")
W = Matrix("w")
Y = Matrix("y")

# MatMulレイヤの順伝搬、逆伝搬は以下の通りです。

# $${{Y}}={{X}}\cdot{{W}}$$

# $${{X.spartial('L')}}={{Y.spartial("L")}}\cdot {{W}}^\mathrm{T}$$

# $${{W.spartial('L')}}={{X}}^\mathrm{T}\cdot {{Y.spartial("L")}}$$

# まずは、MatMalレイヤの動作を確認します。形状を定義します。
N, T, L, M = 20, 3, 3, 4

# ここで$N$はバッチ数、$T$はタイムステップ数、$L$は入力の次元、$M$は出力の次元です。
import numpy as np  # isort:skip

from ivory.core.model import sequential  # isort:skip
from ivory.common.util import convert_one_hot  # isort:skip

net_mat = [("input", L), ("matmul", M, "softmax_cross_entropy")]
model_mat = sequential(net_mat)
mat = model_mat.layers[0]

# ランダムな入力を作成します。
x = np.random.randint(0, L, N)
t = np.random.randint(0, M, N)
# MatMulレイヤへ入力するために、one-hot表現に変換します。
model_mat.set_data(convert_one_hot(x, L), t)
model_mat.forward()
print(model_mat.loss)
# 逆伝搬を比較します。
model_mat.backward()
print(mat.W.g)  # type:ignore
# 数値微分による勾配と比較します。
print(model_mat.numerical_gradient(mat.W.variable))  # type:ignore

# 時系列データを確かめます。
xs = np.random.randint(0, L, (N, T))
ts = np.random.randint(0, M, (N, T))
# MatMulレイヤへ入力するために、one-hot表現に変換します。
model_mat.set_data(convert_one_hot(xs, L), ts)
model_mat.forward()
print(model_mat.loss)
# 逆伝搬を比較します。
model_mat.backward()
print(mat.W.g)  # type:ignore
# 数値微分による勾配と比較します。
print(model_mat.numerical_gradient(mat.W.variable))  # type:ignore

# 次にEmbeddingレイヤを確かめます。まずは時系列でないデータです。
net_em = [("input", L), ("embedding", M, "softmax_cross_entropy")]
model_em = sequential(net_em)
em = model_em.layers[0]
# Embeddingレイヤへは、one-hot表現ではなく、ミニバッチのデータごとにスカラー値を与えます
# 。
model_em.set_data(x, t)
# 両者を比較するために重みを同じ値に設定します。変数の割り当てを変えたらモデルをビルドし
# ます。
em.share_weight_variables(mat)
model_em.build()
# 順伝搬を行います。
model_em.forward()
print(model_em.loss)
# 逆伝搬を比較します。
model_em.backward()
print(em.W.g)  # type:ignore
print(model_em.numerical_gradient(em.W.variable))  # type:ignore

# 次に時系列データです。
model_em.set_data(xs, ts)
model_em.forward()
print(model_mat.loss)
print(mat.y.d[0])
print(em.y.d[0])
# 逆伝搬を比較します。
model_em.backward()
print(mat.W.g)  # type:ignore
# 数値微分による勾配と比較します。
print(model_mat.numerical_gradient(mat.W.variable))  # type:ignore
# 念のため比較します。
grad = np.zeros_like(em.W.d)  # type:ignore
np.scatter_add(grad, em.x.d, em.y.g)  # np.add.at
print(grad)
grad = np.zeros_like(em.W.d)  # type:ignore
for t in range(em.x.d.shape[1]):
    np.scatter_add(grad, em.x.d[:, t], em.y.g[:, t])  # np.add.at
print(grad)


# 以下に実装コードを示します。

# {{ from ivory.layers.core import MatMul }}
# ##Code <code>MatMul</code>クラス
# {{ MatMul # inspect }}

# {{ from ivory.layers.embedding import Embedding }}
# ##Code <code>Embedding</code>クラス
# {{ Embedding # inspect }}
