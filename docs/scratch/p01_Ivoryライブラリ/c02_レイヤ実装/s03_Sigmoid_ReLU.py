# ## Sigmoid/ReLU

# ### ReLU レイヤ

# ReLUレイヤの計算式は以下の通りです。

# $$ y = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases} $$

# $$ \frac{\partial y}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
# $$

# 順伝搬から見ていきます。

# {{ from  ivory.layers.activation import Relu }}
# ##Code <code>Relu.forward</code>メソッド
# {{ Relu.forward # inspect }}

# 「ゼロから作るDeep Learning」と同じですが、引数と戻り値がないことに着目します。
# `self.x.d`はこのレイヤの入力のデータでした。この値自体は前段での順伝搬ですでに計算され
# ています。その計算結果をここでそのまま使っています。またゼロ以下の値をゼロでマスクした
# データを新たに`self.y.d`すなわちこのレイヤの出力のデータにセットしています。この値は後
# 段のレイヤにおいて、`self.x.d`として再利用されます。

# 逆伝搬も見てみます。

# ##Code <code>Relu.backward</code>メソッド
# {{ Relu.backward # inspect }}

# 今度はデータではなく、勾配を伝搬させています。ここで設定された`self.x.g`は前段のレイヤ
# において、`self.y.g`として再利用されます。

# ### Sigmoid レイヤ

# Sigmoidレイヤの計算式は以下の通りです。

# $$y = \frac{1}{1 + \exp(-x)}$$

# $$ \frac{\partial y}{\partial x} = y(1 - y)$$

# 少し寄り道をして、$\partial y/\partial x$ が本当に上式のようになるのか、SymPyを使って計
# 算してみます。

import sympy as sp  # isort:skip

x, y, z = sp.symbols("x y z")
y_x = 1 / (1 + sp.exp(-x))
dy_x = sp.diff(y_x, x)
dy_x

# ${{z}}={{sp.exp(-x)}}$とすると、

dy_z = dy_x.subs(sp.exp(-x), z)
dy_z

# ここで、 ${{y}} = 1/(z+1)$ だから、$z$ で解いて、元の式に代入すると、

sol = sp.solve(y - 1 / (z + 1), z)[0]
sp.simplify(dy_z.subs(z, sol))

# 確かにあっています。実装コードは以下の通りです。

# {{ from ivory.layers.activation import Sigmoid }}
# ##Code <code>Sigmoid.forward</code> および <code>Sigmoid.backward</code>メソッド
# {{ [Sigmoid.forward, Sigmoid.backward] # inspect }}
