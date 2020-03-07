# ## SoftmaxCrossEntropy

# ### 定式化

# ソフトマックス関数と交差エントロピー誤差を合わせたSoftmaxCrossEntropyレイヤを実装します
# 。ソフトマックス関数は、

# $$ y_k = \frac{\exp(x_k)}{\sum_i\exp(x_i)} $$

# 交差エントロピー誤差は、

# $$ L = -\sum_k \delta_{tk}\log y_k $$

# です。ここで$k$は多クラス分類するときのクラス番号に相当します。

# -hide
from ivory.utils.latex import Matrix

X = Matrix("x")
Y = Matrix("y")
T = Matrix("t")

# まずは、形状を定義します。
N, T, M = 2, 3, 4

# ここで$N$はバッチ数、$T$はタイムステップ数、$M$は出力の次元、すなわち、分類するクラス数で
# す。形状の確認を行います。ソフトマックス関数の入力（スコア）を${{X}}$、出力（確率）
# を${{Y}}$、また、交差エントロピー誤差のターゲットを${{T}}$、出力を$L$とします。

# 時系列データでない場合

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X}}$ |$(N, M)$ | $({{N}}, {{M}})$  |
# |${{Y}}$ |$(N, M)$ | $({{N}}, {{M}})$  |
# |$\mathbf{T}$ |$(N,)$ | $({{N}},)$  |
# |$L$ |$()$ | $()$  |
# ~~~

# 時系列データの場合

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X}}$ |$(N, T, M)$ | $({{N}}, {{T}}, {{M}})$  |
# |${{Y}}$ |$(N, T, M)$ | $({{N}}, {{T}}, {{M}})$  |
# |$\mathbf{T}$ |$(N, T)$ | $({{N}}, {{T}})$  |
# |$L$ |$()$ | $()$  |
# ~~~

# ### 時系列データでない場合

# 入力を乱数で発生させます。
import numpy as np  # isort:skip

x = np.random.randn(N, M)
x
# ソフトマックス関数では、オーバーフロー対策のため、バッチデータ(軸1)ごとに最大値を引きま
# す。2次元配列を維持するように、`keepdims=True`とします。
x.max(axis=1, keepdims=True)
# 最大値を引いたものに指数関数を適用します。結果は0から1の範囲に収まります。
exp_x = np.exp(x - x.max(axis=1, keepdims=True))
exp_x
# 次にバッチデータ(軸1)ごとの和で正規化します。これが、ソフトマックス関数の出力になります
# 。
y = exp_x / exp_x.sum(axis=1, keepdims=True)
y
# 当然、次は値が1の配列になります。
y.sum(axis=1)

# 次に、交差エントロピー誤差を求めます。ターゲットは、バッチ数分だけ正解ラベルが並んだベ
# クトルです。
t = np.random.randint(0, M, N)
t
# 例えば、上の例は、バッチデータ0の正解ラベルが{{t[0]}}で、バッチデータ1の正解ラベル
# が{{t[1]}}であることを示します。

# 交差エントロピー誤差はターゲットの位置にあるデータを取り出すことに相当するので、以下の
# ように実装できます。
y_ = y[np.arange(N), t]
y_
# あとは対数の和を取りますが、無限小に発散することを防ぐために微小な値を付加します。また
# 、バッチ数によらない結果を得るために、バッチ数で除算します。
-np.sum(np.log(y_ + 1e-7)) / N

# 「ゼロから作るDeep Learning」の実装と比較します。
from ivory.utils.repository import import_module  # isort:skip

layers = import_module("scratch2/common.layers")
s = layers.SoftmaxWithLoss()
s.forward(x, t)


# 以上が、順伝搬になり、上のスカラー値が損失関数の値になります。

# 逆伝搬は、ソフトマックス関数と交差エントロピー誤差を合わせたレイヤの勾配が次式で与えら
# れることを天下り的に認めたうえで、数値微分によって正しいことを確認します。

# $$ \partial L/x_k = y_k - \delta_{tk} $$
dx = y.copy()
dx[np.arange(N), t] -= 1
dx / N

# 「ゼロから作るDeep Learning」の実装と比較します。
s.backward()


# `SoftmaxCrossEntropy`クラスの実装を確認しておきます。

# {{ from ivory.layers.loss import SoftmaxCrossEntropy }}
# ##Code <code>SoftmaxCrossEntropy</code>クラスの定義
# {{ SoftmaxCrossEntropy # inspect }}

# 実際にインスタンスを作成します。
from ivory.core.model import sequential  # isort:skip

net = [("input", M), "softmax_cross_entropy"]
model = sequential(net)
layer = model.layers[0]
layer.parameters
# 入力とターゲットを代入し、順伝搬を行ってみます。
model.set_data(x, t)
model.forward()
print(model.loss)
# 逆伝搬を求めてみます。
model.backward()
print(layer.x.g)
# 数値微分による勾配と比較してみます。
print(model.numerical_gradient(layer.x.variable))
# 正しいことが確認できました。

# ### 時系列の場合

# 入力を乱数で発生させます。
x = np.random.randn(N, T, M)
t = np.random.randint(0, M, (N, T))
model.set_data(x, t)
model.forward()
print(model.loss)
# 逆伝搬を求めてみます。
model.backward()
print(layer.x.g)
# 数値微分による勾配と比較してみます。
print(model.numerical_gradient(layer.x.variable))
# 正しいことが確認できました。

# 「ゼロから作るDeep Learning」の実装と比較します。
from ivory.utils.repository import import_module  # isort:skip

layers = import_module("scratch2/common.time_layers")
s = layers.TimeSoftmaxWithLoss()
print(s.forward(x, t))
print(s.backward())
