# ## RNN

# 「ゼロから作るDeep Learning ❷」ではT個のRNNレイヤをTime RNNレイヤとして実装しますが
# 、Ivoryライブラリでは、これをまとめてRNNレイヤとして実装します。

# -hide
from ivory.utils.latex import Matrix, Vector

N, n, m = 3, 3, 4
Y = Matrix("y", N, m)
X = Matrix("x", N, n)
W = Matrix("w", n, m)
U = Matrix("u", m, m)
B = Vector("b", m)

# RNNレイヤの入力と出力の関係は以下のようになります。{{ import sympy as sp }}

# $${{Y.s}}_t = {{X.s}}_t\cdot{{W.s}} + {{Y.s}}_{t-1}\cdot{{U.s}} + {{B.s}}$$

# 動作を確かめます。形状（バッチ数、タイムステップ数、入力の次元、出力の次元）を以下の通
# りとします。
from ivory.core.model import sequential  # isort:skip

N, T, L, M = 2, 10, 3, 4

net = [("input", L), ("rnn", M), ("softmax_cross_entropy")]
model = sequential(net)
rnn = model.layers[0]
print(rnn)
# 数値部分のためにビット精度を64ビットにします。
rnn.dtype = 'float64'
for p in rnn.parameters:
    print(p)

# レイヤパラメータを設定します。
from ivory.common.context import np  # isort:skip

w = rnn.W.variable.data = np.random.randn(L, M)  # type:ignore
u = rnn.U.variable.data = np.random.randn(M, M)  # type:ignore
b = rnn.b.variable.data = np.random.randn(M)  # type:ignore
# ランダムな入力を作成します。
x = np.random.randn(N, T, L)
t = np.random.randint(0, M, (N, T))
model.set_data(x, t)
# 順伝搬します。
model.forward()
print(rnn.y.d[:, :2])
# バッチ数分の内積値が出力されました。上記が正しいか、確かめます。
y = np.zeros(M)
for xt in x[0, :2]:
    y = np.tanh(xt @ w + y @ u + b)
    print(y)
print()
y = np.zeros(M)
for xt in x[1, :2]:
    y = np.tanh(xt @ w + y @ u + b)
    print(y)

# 隠れ状態$\mathbf{h}$は最後の出力を保持します。
print(rnn.h.d)  # type:ignore
# 逆伝搬を検証するために、数値微分による勾配確認を行います。
model.forward()
model.backward()
for v in model.grad_variables:
    print(v.parameters[0].name, model.gradient_error(v))
# 一致した結果が得られました。

# 実装は以下のとおりです。

# {{ from ivory.layers.recurrent import RNN }}
# ##Code <code>RNN</code>クラス
# {{ RNN # inspect }}
