# ## Affine

# ### 定式化

# -hide
from ivory.utils.latex import Matrix, Vector

N, n, m = 3, 2, 4
Y = Matrix("y", N, m)
X = Matrix("x", N, n)
W = Matrix("w", n, m)
B = Vector("b", m)

# Affineレイヤの入力と出力の関係は以下のようになります。{{ import sympy as sp }}
# まずは、時系列データでない場合を考えます。

# $${{Y.s}} = {{X.s}}\cdot{{W.s}} + {{B.s}}$$

# ここでバッチ数が$N$、入力ノード数が$n$、出力ノード数が$m$のとき、各行列の次元は次のよう
# になります。

# $$ {{Y.s}}: (N, m),\ {{X.s}}: (N, n),\ {{W.s}}: (n, m),\ {{B.s}}: (m,) $$

# 具体的に、$N={{N}}, n={{n}}, m={{m}}$ の時を書き下します。

# $${{Y}}={{X}}\cdot{{W}} \\ +{{sp.ones(N, 1)}}\cdot{{B}}$$

# ここでは、 ${{X.spartial('L', fold=True)}}$ を導出します。${{W.spartial('L',
# fold=True)}}$、${{B.spartial('L', fold=True)}}$も同様です。

# スカラー関数が分子で、ベクトル・行列が分母のとき、その微分の次元は分母のそれに等しくな
# ります。例えば、入力のバッチ数が{{N}}、入力のノード数が{{n}}のとき、

# $${{X.spartial("L")}} = {{X.partial("L") }}$$

# となります。連鎖率、

# $$ \frac{\partial L}{\partial x_{ij}}  = \sum_{i'j'}\frac{\partial L}{\partial
# y_{i'j'}} \frac{\partial y_{i'j'}}{\partial x_{ij}} $$

# において、

# $$ \frac{\partial y_{i'j'}}{\partial x_{ij}} = \delta_{ii'}w_{jj'} $$

# より、

# $$ \frac{\partial L}{\partial x_{ij}} = \sum_{j'}\frac{\partial L}{\partial y_{ij'}}
# w_{jj'} \rightarrow {{X.spartial("L")}} = {{Y.spartial('L')}}\cdot {{W.s}}^\mathrm{T}
# $$

# となります。他もまとめると、

# $$ {{X.spartial("L")}} = {{Y.spartial('L')}}\cdot {{W.s}}^\mathrm{T} $$

# $$ {{W.spartial('L')}} = {{X.s}}^\mathrm{T}\cdot {{Y.spartial("L")}}$$

# $$ {{B.spartial('L')}} = \mathbf{1}\cdot{{Y.spartial("L")}}$$

# 形状を確認しておきます。時系列データでない場合、
N, L, M = 2, 3, 4

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X.s}}$ |$(N, L)$ | $({{N}},{{L}})$  |
# |${{W.s}}$ |$(L, M)$ | $({{L}}, {{M}})$  |
# |${{B.s}}$ |$(M,)$ | $({{M}},)$  |
# |${{Y.s}}$ |$(N, M)$ | $({{N}}, {{M}})$  |
# ~~~

# 勾配確認を行っていきます。損失パラメータの`grad_variables`は勾配が計算された変数を返す
# イタレータです。
import numpy as np  # isort:skip
from ivory.core.model import sequential  # isort:skip

net = [("input", L), ("affine", M, "softmax_cross_entropy")]
model = sequential(net)
# -
x = np.random.randn(N, L)
t = np.random.randint(0, M, N)
model.set_data(x, t)
model.forward()
model.backward()
# -
for v in model.grad_variables:
    print(v, model.gradient_error(v))
    print(v.grad)
    print(model.numerical_gradient(v))


# ### 時系列データの場合

# 形状を確認おきます。
N, T, L, M = 2, 3, 4, 5

# ~~~markdown
# |パラメータ  |形状  |具体例  |
# |---|---|---|
# |${{X.s}}$ |$(N, T, L)$ | $({{N}},{{T}},{{L}})$  |
# |${{W.s}}$ |$(L, M)$ | $({{L}}, {{M}})$  |
# |${{B.s}}$ |$(M,)$ | $({{M}},)$  |
# |${{Y.s}}$ |$(N, T, M)$ | $({{N}},{{T}}, {{M}})$  |
# ~~~

net = [("input", L), ("affine", M, "softmax_cross_entropy")]
model = sequential(net)
affine = model.layers[0]
# 時系列データの場合でも順伝搬ではこれまで通りです。
x = np.random.randn(N, T, L)
t = np.random.randint(0, M, (N, T))
model.set_data(x, t)
model.forward()
print(affine.y.d)
# 確かめてみます。
print(x[:, 0] @ affine.W.d)  # type:ignore
print(x[:, 1] @ affine.W.d)  # type:ignore
# 逆伝搬を確かめるために、数値微分してみます。入力$\mathbf{X}$です。
print(model.numerical_gradient(affine.x.variable))
# これは、これまでの式が使えます。
model.backward()
dy = affine.y.g
print(dy @ affine.W.d.T)  # type:ignore
# 重みに関しては、前述の式通りでは計算ができません。
print(model.numerical_gradient(affine.W.variable))  # type:ignore
# 同じ重みが複数回使われるので、勾配は加算されます。確かめてみます。
print(sum(x[:, k].T @ dy[:, k] for k in range(T)))
# 形状確認をすることによって、次式が正しい値を与えることが分かります。
print(np.tensordot(x, dy, axes=[(0, 1), (0, 1)]))

# バイアスについても確認します。
print(model.numerical_gradient(affine.b.variable))  # type:ignore
print(dy.sum(axis=(0, 1)))

# 実際にレイヤを使って確かめてみます。
for v in model.grad_variables:
    print(v, model.gradient_error(v))
    print(v.grad)
    print(model.numerical_gradient(v))

# 実装コードの確認しておきます。

# {{ from ivory.layers.affine import Affine }}
# ##Code <code>Affine</code>クラス
# {{ Affine # inspect }}
