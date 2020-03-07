# ## SigmoidCrossEntropy

# ### 定式化

# シグモイド関数と交差エントロピー誤差を合わせたSigmoidCrossEntropyレイヤを実装します。シ
# グモイド関数は、

# $$ y = \frac1{1+\exp(-x)} $$

# 交差エントロピー誤差は、

# $$ L = -\delta_{t0}\log (1-y) - \delta_{t1}\log y$$

# です。ここで、$\delta$はクローネッカーのデルタで、$i=j$のとき$\delta_{ij}=1$、$i\ne
# j$のとき$\delta_{ij}=0$となります。

# -hide
from ivory.utils.latex import Matrix

X = Matrix("x")
Y = Matrix("y")
T = Matrix("t")

# まずは、形状を定義します。
N, T = 3, 4

# ここで$N$はバッチ数、$T$はタイムステップ数です。形状の確認を行います。シグモイド関数の入力（スコア）
# を${{X}}$、出力（確率）を${{Y}}$、また、交差エントロピー誤差のターゲットを${{T}}$、出力
# を$L$とします。

# 時系列データでない場合、

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X}}$ |$(N,)$ | $({{N}},)$  |
# |${{Y}}$ |$(N,)$ | $({{N}},)$  |
# |$\mathbf{T}$ |$(N,)$ | $({{N}},)$  |
# |$L$ |$()$ | $()$  |
# ~~~

# 時系列データの場合、

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X}}$ |$(N, T)$ | $({{N}}, {{T}})$  |
# |${{Y}}$ |$(N, T)$ | $({{N}}, {{T}})$  |
# |$\mathbf{T}$ |$(N, T)$ | $({{N}}, {{T}})$  |
# |$L$ |$()$ | $()$  |
# ~~~

# となります。

# ### 時系列データでない場合

# まず、時系列データでない場合を考え、入力を乱数で発生させます。
import numpy as np  # isort:skip

x = np.random.randn(N)
x

# シグモイド関数の出力を求めます。
y = 1 / (1 + np.exp(-x))
y
# 次に、交差エントロピー誤差を求めます。ターゲットは、バッチ数分だけ正解ラベルが並んだベ
# クトルです。
t = np.random.randint(0, 2, N)
t

# 例えば、上の例は、バッチデータ0の正解ラベルが{{t[0]}}で、バッチデータ1の正解ラベル
# が{{t[1]}}であることを示します。交差エントロピー誤差の式に忠実に書くと、
[-np.log(yi) if ti else -np.log(1 - yi) for yi, ti in zip(y, t)]
# となります。forループを使わないで書くと、
-np.log(np.c_[1 - y, y][np.arange(N), t])
# となります。発散を防ぐための微小値の加算とバッチ数分の平均化をすれば、交差エントロピー
# 誤差が、以下のように求まります。
-np.sum(np.log(np.c_[1 - y, y][np.arange(N), t] + 1e-7)) / N

# 「ゼロから作るDeep Learning」の実装と比較します。
from ivory.utils.repository import import_module  # isort:skip

layers = import_module("scratch2/common.layers")
s = layers.SigmoidWithLoss()
s.forward(x, t)

# 以上が、順伝搬になり、上のスカラー値が損失関数の値になります。

# 逆伝搬は、シグモイド関数と交差エントロピー誤差を合わせたレイヤの勾配が次式で与えられる
# ことを天下り的に認めたうえで、数値微分によって正しいことを確認します。

# $$ \partial L/x = y - t $$

dx = (y - t) / N
dx
# 「ゼロから作るDeep Learning」の実装と比較します。
s.backward()

# `SigmoidCrossEntropy`クラスの実装を確認しておきます。

# {{ from ivory.layers.loss import SigmoidCrossEntropy }}
# ##Code <code>SigmoidCrossEntropy</code>クラスの定義
# {{ SigmoidCrossEntropy # inspect }}

# 実際にインスタンスを作成します。
from ivory.core.model import sequential  # isort:skip

net = [("input", ()), "sigmoid_cross_entropy"]
model = sequential(net)
layer = model.layers[0]
layer.parameters

# 変数を設定した後、入力とターゲットを代入します。
layer.set_variables()
layer.set_data(x, t)
# 順伝搬を行います。
model.forward()
# 逆伝搬を行います。
model.backward()
layer.x.g
# さて、この勾配が正しいかは、数値微分による勾配確認によって検証できます。入力データの
# 第1要素を少しだけずらして、損失を求めます。
epsilon = 1e-4
layer.x.d[0] += epsilon
model.forward()
plus = model.loss
# 逆方向にずらします。
layer.x.d[0] -= 2 * epsilon
model.forward()
minus = model.loss
# 勾配は次式で得られます。
print((plus - minus) / (2 * epsilon))
# これまでの結果に一致しています。入力をもとに戻しておきます。
layer.x.d[0] += epsilon

# ある変数の全ての要素について数値微分による勾配を求めるメソッド`numerical_gradient`が用
# 意されています。

print(model.numerical_gradient(layer.x.variable))
# 正しいことが確認できました。

# ### 時系列データの場合

# 入力を乱数で発生させます。
x = np.random.randn(N, T)
t = np.random.randint(0, 2, (N, T))

layer.set_data(x, t)
# 順伝搬を行います。
model.forward()
print(layer.loss.d)  # type:ignore
# 逆伝搬を行います。
model.backward()
print(layer.x.g)
# 数値微分で確かめてみます。
print(model.numerical_gradient(layer.x.variable))

# 「ゼロから作るDeep Learning」の実装と比較します。
from ivory.utils.repository import import_module  # isort:skip

layers = import_module("scratch2/common.time_layers")
s = layers.TimeSigmoidWithLoss()
print(s.forward(x, t))
print(s.backward())
