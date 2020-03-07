# ## 変数

# Ivoryライブラリでは、入出力のデータやレイヤの重みパラメータなどの実データを変
# 数`Variable`クラスで扱います。

# 例を見てみます。

from ivory.core.variable import Variable

v = Variable((3, 4))
v

# 第1引数には形状を指定します。ここで形状にはミニバッチ学習におけるバッチサイズを含めませ
# ん。バッチサイズはあくまでも訓練にかかわるもので、変数自体がその情報を持つ必要はないと
# いう考えに基づいています。

# `Variable`インスタンスの持つ属性を確認します。

vars(v)

# `parameters`属性は後述する`Parameter`インスタンスの参照です。変数は、複数
# の`Parameter`で共有されることがあるので、リストになっています。いちばん身近な例は、2つ
# のレイヤを接続するときです。このとき最初のレイヤの出力と次のレイヤの入力は同じ変数を共
# 有します。現在は変数`v`は孤立した状態なので、値が空リストとなっています。`data`属性
# と`grad`属性はデータと勾配を保持するNumPyもしくはCuPyの`ndarray`です。現在は値が割り当
# てられていません。`init`属性は変数の値を初期化する際に呼ばれる関数です。

# 値を入力するには、直接`data`属性、`grad`属性にアクセスします。

import numpy as np  # isort:skip

v.data = np.random.randn(*v.shape)
v.grad = np.zeros(v.shape)
print("v.data = \n", v.data)
print("v.grad = \n", v.grad)
