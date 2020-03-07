# ## LSTM

# 「ゼロから作るDeep Learning ❷」ではT個のLSTMレイヤをTime LSTMレイヤとして実装しますが
# 、Ivoryライブラリでは、これをまとめてLSTMレイヤとして実装します。

# 動作を確かめます。形状（バッチ数、タイムステップ数、入力の次元、出力の次元）を以下の通
# りとします。
from ivory.core.model import sequential  # isort:skip

N, T, L, M = 3, 10, 3, 4

net = [("input", L), ("lstm", M), ("softmax_cross_entropy")]
model = sequential(net)
lstm = model.layers[0]
print(lstm)
# 数値部分のためにビット精度を64ビットにします。パラメータはRNNレイヤと同じです。
lstm.dtype = 'float64'
for p in lstm.parameters:
    print(p)

# レイヤパラメータを設定します。
from ivory.common.context import np  # isort:skip

w = lstm.W.variable.data = np.random.randn(L, 4 * M)  # type:ignore
u = lstm.U.variable.data = np.random.randn(M, 4 * M)  # type:ignore
b = lstm.b.variable.data = np.random.randn(4 * M)  # type:ignore
# ランダムな入力を作成します。
x = np.random.randn(N, T, L)
t = np.random.randint(0, M, (N, T))
model.set_data(x, t)
# 順伝搬します。
model.forward()
print(lstm.y.d[:, :2])


# バッチ数分の内積値が出力されました。上記が正しいか、確かめます。
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for i in range(2):
    y = np.zeros(M)
    c = 0
    for xt in x[i, :2]:
        a = xt @ w + y @ u + b
        f, i, o, g = a[:M], a[M : 2 * M], a[2 * M : 3 * M], a[3 * M :]
        f = sigmoid(f)
        i = sigmoid(i)
        o = sigmoid(o)
        g = np.tanh(g)
        c = f * c + g * i
        y = o * np.tanh(c)
        print(y)
    print()

# 隠れ状態`h`は最後の出力を保持します。
print(lstm.h.d)  # type:ignore
# 逆伝搬を検証するために、数値微分による勾配確認を行います。
model.forward()
model.backward()
for v in model.grad_variables:
    print(v.parameters[0].name, model.gradient_error(v))
# 一致した結果が得られました。

# 実装は以下のとおりです。

# {{ from ivory.layers.recurrent import LSTM }}
# ##Code <code>LSTM</code>クラス
# {{ LSTM # inspect }}
