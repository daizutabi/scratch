# ## MatMulMean

# -hide
from ivory.utils.latex import Matrix

X = Matrix("x")
W = Matrix("w")
Y = Matrix("y")


# 「ゼロから作るDeep Learning ❷」3章のword2vecでは、2つのMatMulレイヤが重みを共有しながら
# 、両者の出力を平均を出力していました。ここでは、形状確認と勾配確認の手法を使って一つの
# レイヤとして実装します。

# まずは、形状を定義します。
N, I, L, M = 2, 3, 4, 5

# ここで$N$はバッチ数、$I$は入力数、$L$は入力の次元、$M$は出力の次元です。

# 通常のMatMulレイヤの順伝搬、逆伝搬は以下の通りです。

# $${{Y}}={{X}}\cdot{{W}}$$

# $${{X.spartial('L')}}={{Y.spartial("L")}}\cdot {{W}}^\mathrm{T}$$

# $${{W.spartial('L')}}={{X}}^\mathrm{T}\cdot {{Y.spartial("L")}}$$

# 実験のために、乱数行列を作成します。
import numpy as np  # isort:skip

x = np.random.randn(N, I, L)
w = np.random.randn(L, M)

# まずは、入力を別々にして、MatMulレイヤの計算をひとつずつ行った場合を示します。
xs = x.transpose(1, 0, 2)
sum(xi @ w for xi in xs) / len(xs)

# 各パラメータの形状を確認しておきます。

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |$\mathbf X$ |$(N, I, L)$ | $({{N}}, {{I}}, {{L}})$  |
# |$\mathbf W$ |$(L, M)$ | $({{L}}, {{M}})$  |
# |$\mathbf X\cdot\mathbf W$ |$(N, I, M)$ | $({{N}}, {{I}}, {{M}})$  |
# |$\mathbf Y$ |$(N, M)$ | $({{N}}, {{M}})$  |
# ~~~

# ${{X}}$と${{W}}$の内積をとった後に${{Y}}$の形状に合わせるには、軸1で和を取ります。

np.sum(x @ w, axis=1) / x.shape[1]

# 順伝搬は以上です。ひとまず、ここまでを`MatMulMean`レイヤとして実装します。イニシャライ
# ザに与える形状は、$(I, L, M)$とします。
from ivory.core.layer import Layer  # isort:skip


class MatMulMean(Layer):
    def init(self):
        self.W = self.add_weight(self.shape[1:]).randn()

    def forward(self):
        self.y.d = np.sum(self.x.d @ self.W.d, axis=1) / self.shape[0]


# 動作を確認しておきます。
layer = MatMulMean((I, L, M))
print(layer)
print(layer.x)
print(layer.W)  # type:ignore
print(layer.y)
# -
vs = layer.set_variables()
vs[0].data = x
vs[2].data = w
layer.forward()
layer.y.d

# 次に逆伝搬を実装します。ここで勾配確認のテクニックを使います。今作ったMatMulMeanレイヤ
# に損失関数を繋げます。
from ivory.core.model import Model  # isort:skip
from ivory.layers.loss import SoftmaxCrossEntropy  # isort:skip

loss_layer = SoftmaxCrossEntropy((M,))
loss_layer.set_input_layer(layer)
model = Model([loss_layer.loss]).build()

# ターゲットを乱数で生成します。
loss_layer.t.variable.data = np.random.randint(0, M, N)
# 数値微分による勾配を求めます。まずは、${{X.spartial('L', True)}}$です。
model.numerical_gradient(layer.x.variable)
# 入力数の数だけ、同じ勾配が分配されていることが分かります。

# MatMulMeanレイヤの出力の勾配を求めます。
try:
    model.backward()
except AttributeError:  # まだ、backwardメソッドを定義していないため
    pass
layer.y.g
# これを使ってとにかく単純なMatMulレイヤと同様の計算をしてみます。
dx = (layer.y.g @ layer.W.d.T) / layer.shape[0]  # type:ignore
dx
# 軸0で`repeat`した後、`reshape`するのがよさそうです。
np.repeat(dx, layer.shape[0], axis=0).reshape(*layer.x.d.shape)

# 数値微分と一致しました。次は、${{W.spartial('L', True)}}$です。
model.numerical_gradient(layer.W.variable)  # type:ignore
# MatMulレイヤと同様の計算をする前に、形状を確認しておきます
# 。{{tmp=Y.spartial("L",True)}}

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{Y}}$ |$(N, M)$ | $({{N}}, {{M}})$  |
# |$\mathbf X$ |$(N, I, L)$ | $({{N}}, {{I}}, {{L}})$  |
# |${{X}}^\mathrm{T}$ |$(L, I, N)$ | $({{L}}, {{I}}, {{N}})$  |
# |${{X}}^\mathrm{T}\cdot {{tmp}}$ |$(L, I, M)$ | $({{L}}, {{I}}, {{M}})$  |
# |$\mathbf W$ |$(L, M)$ | $({{L}}, {{M}})$  |
# ~~~

# これより、必要な計算が明らかになります。
np.sum(layer.x.d.T @ layer.y.g, axis=1) / layer.shape[0]
# 数値微分と一致しました。`backward`メソッドを追加した実装は以下の通りです。

# {{ from ivory.layers.core import MatMulMean }}
# ##Code <code>MatMulMean</code>クラスの実装
# {{ MatMulMean # inspect }}

# 動作の確認を行います。
from ivory.core.model import sequential  # isort:skip

net = [("input", I, L), ("matmulmean", M, "softmax_cross_entropy")]
model = sequential(net)
mat = model.layers[0]
mat.x

model.set_data(np.random.randn(N, I, L), np.random.randint(0, M, N))
model.forward()
model.backward()

for var in model.grad_variables:
    error = model.gradient_error(var)
    print(var.parameters[0].name, f"{error:.04e}")
# 以上のように、形状確認と勾配確認によって、新規にレイヤを実装することができました。
