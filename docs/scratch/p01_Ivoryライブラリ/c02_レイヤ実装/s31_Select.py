# ## Select

# seq2seqで使うために、時系列データの中から一つのタイムステップを選択する`Select`レイヤ
# を実装します。
from ivory.core.model import sequential

N, T, L, M = 2, 10, 4, 3

net = [("input", L), ("rnn", M, "select", "softmax_cross_entropy")]
model = sequential(net)
rnn, select = model.layers[:2]
rnn.dtype = "float64"
model.init()
model.layers

# `Select`レイヤを通過することで時系列データの一つが選択されます。デフォルトは時系
# 列データの最後のタイムステップです。
import numpy as np  # isort:skip

x = np.random.randn(N, T, L)
t = np.random.randint(0, M, N)
model.set_data(x, t)
model.forward()
print(select.x.d[:, -1])
print(select.y.d)

# 逆伝搬を検証するために、数値微分による勾配確認を行います。
model.forward()
model.backward()
for v in model.grad_variables:
    print(v.parameters[0].name, model.gradient_error(v))
# 一致した結果が得られました。

# 実装は以下のとおりです。

# {{ from ivory.layers.recurrent import Select }}
# ##Code <code>Select</code>クラス
# {{ Select # inspect }}
