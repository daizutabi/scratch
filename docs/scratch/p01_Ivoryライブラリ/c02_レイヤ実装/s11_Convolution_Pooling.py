# ## Convolution/Pooling

# Convolutionレイヤについて整理しておきます。2次元配列を複数チャネル持ったデータをあるバ
# ッチ数まとめて入力します。このデータ形状を、$($バッチ数$N$, チャネル数$C$, 高
# さ$H$,幅$W)$とします。Convolutionレイヤ自体は、パラメータとして3次元フィルタを複数個持
# ち、同じ数だけスカラーのバイアスを持ちます。フィルタの形状を$($フィルタ数$FN$, チャネル
# 数$C$, 高さ$FH$, 幅$FW)$とします。出力形状は$($バッチ数$N$, チャネル数$FN$, 高
# さ$OH$,幅$OW)$となります。ここで、パディング$P$、ストライド$S$としたとき、

# $$ OH = \frac{H+2P-FH}{S}+1 $$

# $$ OW = \frac{W+2P-FW}{S}+1 $$

# です。注目する点は、

# * フィルタは、入力データのチャネル数分をまとめて新たな一つの「画像」を作る
# * 出力のチャネル数はフィルタ数に等しくなる
# * 新しい「画像」のサイズが上述の（$OH$, $OW$）となる

# です。

# ### 4次元配列

# -hide
import re
from itertools import product

import numpy as np
import sympy as sp

from ivory.layers.convolution import Convolution, Pooling


def latex(array):
    if not isinstance(array[0][0], str):
        return sp.latex(sp.Matrix(array))

    def replace(x):
        return re.sub(r"(.)_(\d+)", r"\1_{\2}", x)

    mat = r"\\".join("&".join(replace(x) for x in row) for row in array)
    return r"\left[\begin{matrix}" + mat + r"\end{matrix}\right]"


class Matrix:
    def __init__(self, value, *shape):
        if not shape:
            self.value = np.array(value)
        else:
            prod = product(*map(range, shape))
            sub = ("".join(str(i + 1) for i in x) for x in prod)
            array = ["_".join([value, x]) for x in sub]
            self.value = np.array(array).reshape(shape)

    def _repr_latex_(self):
        if self.value.ndim == 2:
            value = latex(self.value)
        elif self.value.ndim == 3:
            value = latex([[latex(x) for x in self.value]])
        elif self.value.ndim == 4:
            m = [[latex(x) for x in row] for row in self.value]
            value = latex(m)
        return value, {"module": "sympy"}


# Convolutionレイヤへの入力データ$\mathbf{X}$は4次元配列です。データの形状を$($バッチ
# 数$N$, チャネル数$C$, 高さ$H$, 幅$W)$とします。

# -hide
N, C, H, W = 3, 2, 2, 3
X = Matrix("x", N, C, H, W)

# $$\mathbf X = {{X}} $$

# 上記の例では、高さ{{H}}、幅{{W}}の画像が、チャネル数{{C}}（横方向）、バッチ数{{N}}（縦
# 方向）で並んでいると解釈できます。

# ### im2colによる展開


# -hide
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype="object")

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


FH, FW = 2, 2
Xhat = Matrix(im2col(X.value, FH, FW))

# 「ゼロから作るDeep Learning」で導入されている`im2col`関数の動作を確認します。フィルタサ
# イズが、$FH={{FH}}$、$FW={{FW}}$の場合に、`im2col`関数で$\mathbf{X}$を2次元配列に変換し
# た結果$\hat{\mathbf X}$は以下のようになります。

# $$\hat{\mathbf X}={{Xhat}}$$

# $\hat{\mathbf X}$の形状を確認しておきます。列の数は、一つのフィルタの形状$(C,\ FH,\
# FW)$の要素数$C\times FH\times FW$に等しくなります。つまり、ある一つのフィルタによって
# 畳み込まれる要素が一行に並ぶ形になります。今回の場合は、${{C}}\times {{FH}}\times
# {{FW}}={{C*FH*FW}}$です。行方向はどうでしょうか。今は、チャネルあたりの元画像
# が${{H}}\times {{W}}$でフィルタが${{FH}}\times {{FW}}$です。出力画像のサイズの式：

# $$ OH = \frac{H+2P-FH}{S}+1 $$

# $$ OW = \frac{W+2P-FW}{S}+1 $$


# -hide
def output_size(h, w, filter_h, filter_w, stride=1, pad=0):
    oh = (h + 2 * pad - filter_h) // stride + 1
    ow = (w + 2 * pad - filter_w) // stride + 1
    return oh, ow


OH, OW = output_size(H, W, FH, FW)

# を使うと、$OH={{OH}}$、$OW={{OW}}$で、出力される一つの画像の画素数が{{OH * OW}}となりま
# す。これが、バッチ数{{N}}個分繰り返されるので、行数は${{OH * OW}}\times {{N}}= {{OH *
# OW * N}}$となります。このように、一つのフィルタでの畳み込みで、出力画素ひとつが生成さ
# れるので、行数は出力画像の総画素数になります。結果的に、入力の4次元配列$(N,\ C,\ H,\
# W)$は、

# $$(N\times OH\times OW,\ C\times FH\times FW)=({{N*OH*OW}},\ {{C*FW*FH}})$$

# の2次元配列となりました。一つの行が一つのフィルタで変換され、それが出力される画像の総画
# 素数分だけ行方向に並ぶ形状です。

# -hide
FN = 4
F = Matrix("w", FN, C, FH, FW)

# 次にフィルタ$\mathbf W$を見ていきます。フィルタはバッチ数とは関係なく、$($フィルタ
# 数$FN$, チャネル数$C$, 高さ$FH$, 幅$FW)$の4次元配列です。$FN={{FN}}$のとき、

# $$\mathbf W = {{F}}$$

# です。なお、ここでのフィルタ数$FN$が次のレイヤでのチャネル数$C$になります。

# ### Convolutionレイヤの実装
# #### 順伝搬

# {{url = 'deep-learning-from-scratch/blob/master/common/layers.py' }} [「ゼロから作
# るDeep Learning」の実装](https://github.com/oreilly-japan/{{url}})を確認します。

# ~~~python
# def forward(self, x):
#     FN, C, FH, FW = self.W.shape
#     N, C, H, W = x.shape
#     out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
#     out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
#     col = im2col(x, FH, FW, self.stride, self.pad)
#     col_W = self.W.reshape(FN, -1).T
#     out = np.dot(col, col_W) + self.b
#     out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
#     self.x = x
#     self.col = col
#     self.col_W = col_W
#     return out
# ~~~

# 上記のフィルタ$\mathbf W$は以下のように変換されています。

# ~~~python
#     col_W = self.W.reshape(FN, -1).T
# ~~~

# 変換後のフィルタを$\hat{\mathbf W}$とすると、

# -hide
Fhat = Matrix(F.value.reshape(FN, -1).T)

# $$\hat{\mathbf W}={{Fhat}}$$

# です。一つの列が一つのフィルタを平坦化したものに対応し、フィルタ数分だけ列があります。
# すなわち形状は、$(C\times FH\times FW,\ FN)$となります。今回の場合は、$({{C*FH*FW}},\
# {{FN}})$です。

# さて、先の2次元化した入力$\hat{\mathbf X}$と比べてみます。こちらは一つの行が一つのフィ
# ルタに対応していました。

# $$\hat{\mathbf X}={{Xhat}}$$

# 順伝搬においては、2次元化したこれらの配列をAffineレイヤと同じように、

# $$\hat{\mathbf Y} = \hat{\mathbf X}\cdot\hat{\mathbf W} + \mathbf B$$

# とします。（$\mathbf B$はフィルタ数分のスカラー値を持ったバイアスで計算時にはブロードキ
# ャストされます。）形状を再確認しておきます。

# ~~~markdown
# |2次元化した配列  |行数  |列数  |
# |---|---|---|
# |$\hat{\mathbf X}$  |$N\times OH\times OW$  |$C\times FH\times FW$  |
# |$\hat{\mathbf W}$  |$C\times FH\times FW$  |$FN$  |
# |$\hat{\mathbf Y}$  |$N\times OH\times OW$  |$FN$  |
# ~~~

# 結果的に出力$\hat{\mathbf Y}$の形状は、上表のとおりになります。

# 次に、「ゼロから作るDeep Learning」の実装では、上述の計算結果を`out`として、

# ~~~python
#     out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
# ~~~

# としています。`reshape`によって


# $$(N\times OH\times OW,\ FN)\rightarrow(N,\ OH,\ OW,\ FN)$$

# となり、続く`transpose`によって軸の順番が変わり、

# $$(N,\ OH,\ OW,\ FN)\rightarrow(N,\ FN,\ OH,\ OW)$$

# となります。結果的にConvolutionレイヤを通過することによって、

# $$(N,\ C,\ H,\ W)\rightarrow(N,\ FN,\ OH,\ OW)$$

# というふうに、「チャネル数」、「画像サイズ」が変換されます。当然、バッチ数には変化があ
# りません。


# #### 逆伝搬
# 逆伝搬についても見ていきます。 「ゼロから作るDeep Learning」の実装を確認しておきます。

# ~~~python
# def backward(self, dout):
#     FN, C, FH, FW = self.W.shape
#     dout = dout.transpose(0,2,3,1).reshape(-1, FN)
#     self.db = np.sum(dout, axis=0)
#     self.dW = np.dot(self.col.T, dout)
#     self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
#     dcol = np.dot(dout, self.col_W.T)
#     dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
#     return dx
# ~~~

# フィルタの形状が、$(FN,\ C,\ FH,\ FW)$だったのを思い出しましょう。

# 前述のとおり、Convolutionレイヤを通過することによって、$(N,\ FN,\ OH,\ OW)$が伝搬されて
# いるので、逆伝搬における勾配も同じ形状です。Convolutionレイヤの出力の勾配$\partial
# L/\partial \mathbf{Y}$を$\mathbf G$とします。

# -hide
G = Matrix("g", N, FN, OH, OW)

# $$\frac{\partial L}{\partial \mathbf{Y}} = \mathbf G={{G}}$$

# それを、

# ~~~python
#     dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
# ~~~

# とすることによって、形状変換します。

# $$(N,\ FN,\ OH,\ OW)\rightarrow(N,\ OH,\ OW,\ FN)\rightarrow(N\times OH\times OW,\
# FN)$$

# 実際に計算すると、

# -hide
Ghat = Matrix(G.value.transpose(0, 2, 3, 1).reshape(-1, FN))

# $$\frac{\partial L}{\partial \hat{\mathbf{Y}}} = \hat{\mathbf G}={{Ghat}}$$

# となり、順伝搬における

# $$\hat{\mathbf Y}=\hat{\mathbf X}\cdot\hat{\mathbf W} + \mathbf B$$

# と同じ形状になります。次に、パラメータと入力の勾配を求めます。該当部分を再掲します。

# ~~~python
#     self.db = np.sum(dout, axis=0)
#     self.dW = np.dot(self.col.T, dout)
#     dcol = np.dot(dout, self.col_W.T)
# ~~~

# $\partial L/\partial\mathbf{B}$はAffineレイヤと同じように軸0で和を取ります。$\partial
# L/\partial\hat{\mathbf{W}}$と$\partial L/\partial\hat{\mathbf{X}}$はAffineレイヤと同じ
# ように、

# $$\frac{\partial L}{\partial\hat{\mathbf{W}}} = \hat{\mathbf
# X}^\mathrm{T}\cdot\frac{\partial L}{\partial\hat{\mathbf{Y}}}$$

# $$\frac{\partial L}{\partial\hat{\mathbf{X}}} = \frac{\partial
# L}{\partial\hat{\mathbf{Y}}}\cdot\hat{\mathbf W}^\mathrm{T} $$

# です。右辺の形状を確認しておきます。

# ~~~markdown
# |2次元化した配列  |行数  |列数  |
# |---|---|---|
# |$\hat{\mathbf X}$  |$N\times OH\times OW$  |$C\times FH\times FW$  |
# |$\hat{\mathbf W}$  |$C\times FH\times FW$  |$FN$  |
# |$\partial L/\partial \hat{\mathbf Y}$  |$N\times OH\times OW$  |$FN$  |
# ~~~

# 上式の内積の結果は、

# $$ (C\times FH\times FW,\ N\times OH\times OW)\cdot(N\times OH\times OW,\ FN)
# \rightarrow(C\times FH\times FW,\ FN)$$

# $$ (N\times OH\times OW,\ FN)\cdot(FN,\ C\times FH\times FW) \rightarrow(N\times
# OH\times OW,\ C\times FH\times F)$$

# となり、確かに、$\hat{\mathbf W}$と$\hat{\mathbf X}$の形状に一致します。後は、前のレイ
# ヤに逆伝搬できるように、$\mathbf W$と$\mathbf X$の形状に戻すだけです。

# ~~~markdown
# |4次元配列  |形状   |
# |---|---|
# |$\mathbf X$  |$(N,\ C,\ H,\ W)$|
# |$\mathbf W$  |$(FN,\ C,\ FH,\ FW)$|
# ~~~

# $\partial L/\hat{\mathbf{W}}\rightarrow\partial L/\mathbf{W}$については簡単です。形状
# の順番が違うだけなので、転置した後、`reshape`で4次元配列に戻しています:

# ~~~python
#     self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
# ~~~

# $\partial L/\hat{\mathbf{X}}\rightarrow\partial L/\mathbf{X}$については、`im2col`関数
# の逆変換、`col2im`関数が用意されています。逆伝搬の関数の中で、該当する部分は以下です。

# ~~~python
#     dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
# ~~~


# -hide
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros(
        (N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype="object"
    )
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]


# さて、2次元化された入力$\hat{\mathbf{X}}$が、

# $$\hat{\mathbf X}={{Xhat}}$$

# だったことを思い出しましょう。勾配も同じ形状なので、簡単のためにこのまま`im2col`関数に
# 渡してみます。ここで、$x$は入力データではなく、勾配であると読み替えます。すると、4次元
# に戻った勾配$\partial L/\partial \mathbf{X}$が、

# -hide
x = np.array(sp.Matrix(Xhat.value))
x = Matrix(col2im(x, X.value.shape, FH, FW))

# $$\frac{\partial L}{\partial \mathbf{X}} = {{x}}$$

# と計算されます。入力データをもう一度確認します。

# $$\mathbf{X} = {{X}}$$

# 添え字の場所が一致していることが確認できます。また、定数倍になっているのは、文字通り定
# 数倍するのではなく、その要素が出力に伝搬される経路数を表しています。経路ごとに異なった
# 勾配が、その要素の位置で和算されます。畳み込みされる頻度の高い画像中央ほど経路数が多い
# ことが確認できます。

# 以上で、Convolutionレイヤの実装の確認ができました。

# ### Poolingレイヤの実装

# #### 順伝搬

# 「ゼロから作るDeep Learning」の実装を確認します。

# ~~~python
# def forward(self, x):
#     N, C, H, W = x.shape
#     out_h = int(1 + (H - self.pool_h) / self.stride)
#     out_w = int(1 + (W - self.pool_w) / self.stride)
#     col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
#     col = col.reshape(-1, self.pool_h * self.pool_w)
#     arg_max = np.argmax(col, axis=1)
#     out = np.max(col, axis=1)
#     out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
#     self.x = x
#     self.arg_max = arg_max
#     return out
# ~~~

# Convolutionレイヤと同じ入力を考えます。

# $$\mathbf X={{X}}$$

# -hide
PH, PW = 2, 2
col = im2col(X.value, PH, PW)
Xhat = Matrix(col.reshape(-1, PH * PW))

# `im2col`関数を適用した後、`reshape`します。ここで、$PH={{PH}}$、$PW={{PW}}$とします。

# $$\hat{\mathbf X}={{Xhat}}$$

# 配列の形状は以下のようになります。

# ~~~markdown
# |2次元化した配列  |行数  |列数  |
# |---|---|---|
# |$\hat{\mathbf X}$  |$N\times OH\times OW\times C$  |$PH\times PW$  |
# ~~~

# Convolutionレイヤの場合と比べて、`reshape`の結果、チャネル$C$が列から行に移っています
# 。後は軸1で最大値を取った後、`reshape`と`transpose`によって、

# $$N\times OH\times OW\times C\rightarrow(N,\ OH,\ OW,\ C) \rightarrow(N,\ C,\ OH,\
# OW)$$

# となります。このように、Poolingレイヤを通過することで、バッチ数とチャネル数はそのままで
# 、画像のサイズが$OH\times OW$に変更になりました。

# #### 逆伝搬

# 「ゼロから作るDeep Learning」の実装を確認します。

# ~~~python
# def backward(self, dout):
#     dout = dout.transpose(0, 2, 3, 1)
#     pool_size = self.pool_h * self.pool_w
#     dmax = np.zeros((dout.size, pool_size))
#     dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
#     dmax = dmax.reshape(dout.shape + (pool_size,))
#     dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
#     dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
#     return dx
# ~~~

# 前述のとおり、Poolingレイヤを通過することによって、$(N,\ C,\ OH,\ OW)$が伝搬されている
# ので、逆伝搬における勾配も同じ形状です。Poolingレイヤの出力の勾配$\partial L/\partial
# \mathbf{Y}$を$\mathbf G$とします。

# -hide
G = Matrix("g", N, C, OH, OW)

# $$\frac{\partial L}{\partial \mathbf{Y}} = \mathbf G={{G}}$$

# `transpose`によって、中間状態`dout`は、

# $$(N,\ C,\ OH,\ OW) \rightarrow (N,\ OH,\ OW,\ C)$$

# となります。また、`dmax`は、$\hat{\mathbf{X}}$と同じ形状のゼロ配列です。

# -hide
dout = G.value.transpose(0, 2, 3, 1)
dmax = np.zeros((dout.size, PH * PW), dtype="object")

# `dout`は`flatten`されてベクトルになった後、`dmax`に代入されますが、このとき、元々の入
# 力が最大だった列へのみ代入します。

# -hide
dout_ = dout.flatten()
dmax_ = np.array([[k] * PH * PW for k in dout_])


# $$\mathrm{dout'}={{Matrix(np.array([dout_]).T)}},\ \ \mathrm{dmax}={{Matrix(dmax)}} \
# \rightarrow {{Matrix(dmax_)}} $$

# 上の例では、仮想的に`dmax`のすべての列に勾配を代入しています。本来であれば、非ゼロの要
# 素は各行につき一つです。形状を確認しておきます。

# ~~~markdown
# |2次元化した配列  |行数  |列数  |
# |---|---|---|
# |$\mathrm{dmax}$  |$N\times OH\times OW\times C$  |$PH\times PW$  |
# ~~~

# つぎに、2回の`reshape`によって、

# $$(N\times OH\times OW\times C,\ PH\times PW)\rightarrow (N,\ OH,\ OW,\ C,\ PH\times
# PW)\ $$


# $$(N,\ OH,\ OW,\ C,\ PH\times PW)\rightarrow (N\times OH\times OW,\ C\times PH\times
# PW)$$

# と変化します。実際に確認してみます。

# -hide
dcol = dmax_.reshape(dout.shape + (PH * PW,))
dcol = dcol.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)

# $$\mathrm{dcol} = {{Matrix(dcol)}} $$

# 最終的に、`col2im`関数を適用します。

# -hide
x = np.array(sp.Matrix(dcol))
x = Matrix(col2im(x, X.value.shape, PH, PW))

# $$\frac{\partial L}{\partial \mathbf{X}} = {{x}}$$

# ここで同じ添え字の勾配が$PH\times PW$回出現しますが、実際にゼロではないのは一つです。そ
# して、その位置は、Poolingされる範囲で最大の要素がある位置になります。

# 以上でPoolingレイヤの実装の確認ができました。

# ### Ivoryライブラリでの実装

# 実際にIvoryライブラリでの実装を確認します。まずは孤立したレイヤを作成する例を示します。

conv = Convolution((2, 6, 6, 3, 3, 3))  # (C, H, W, FN, FH, FW)
print(conv.x)  # (C, H, W)
print(conv.W)  # (FN, C, FH, FW)
print(conv.b)  # (FN,)
print(conv.y)  # (FN, OH, OW)

# -
pool = Pooling((3, 4, 4, 2, 2))  # (C, H, W, PH, PW)
print(pool.x)  # (C, H, W)
print(pool.y)  # (C, OH, OW)

# 「ゼロから作るDeep Learing」の`SimpleConvNet`を再現します。ここではAffineレイヤを接続す
# るためにFlattenレイヤを間に入れます。
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 1, 10, 10),
    ("convolution", 10, 3, 3, "relu"),
    ("pooling", 2, 2, "flatten"),
    ("affine", 10, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
model = sequential(net)
model.layers

# 例題に合わせるため重みの標準偏差を0.01に設定します。
for v in model.weight_variables:
    if v.parameters[0].name == "W":
        v.data = v.init(std=0.01)  # type:ignore
        print(v.parameters[0].name, v.data.std())  # type:ignore

# ランダムデータで評価してみます。
import numpy as np  # isort:skip

x = np.random.rand(100).reshape((1, 1, 10, 10))
t = np.array([1])

model.set_data(x, t)
model.forward()
model.backward()

for v in model.grad_variables:
    print(v.parameters[0].name, model.gradient_error(v))

# 最後に、実装コードを記載します。

# ##File <code>layers/convolution.py</code>
# {%=/ivory/layers/convolution.py%}
