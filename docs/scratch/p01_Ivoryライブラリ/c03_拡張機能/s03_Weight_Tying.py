# ## 重み共有

# Ivoryライブラリでは重みを共有したときの勾配の加算は自動で行われます。
import numpy as np  # isort:skip
from ivory.core.trainer import sequential  # isort:skip

N, L, M = 3, 2, 3
net = [("input", L), ("affine", L), ("affine", L, "softmax_cross_entropy")]
trainer = sequential(net)
model = trainer.model
m1, m2, s = model.layers
w = np.random.randn(L, L)
m1.W.variable.data = w  # type:ignore
m2.W.variable.data = w.copy()  # type:ignore
# データを作成します。
x = np.random.randn(N, L)
t = np.random.randint(0, L, N)
model.set_data(x, t)
model.forward()
model.backward()
print(m1.W.g)  # type:ignore
print(model.numerical_gradient(m1.W.variable))  # type:ignore

print(m2.W.g)  # type:ignore
print(model.numerical_gradient(m2.W.variable))  # type:ignore
# 勾配を加算してみます。
print(m1.W.g + m2.W.g)  # type:ignore

# 重みを共有し、トレーナーをビルドします。
m2.W.share_variable(m1.W)  # type:ignore
trainer.build()
for v in trainer.optimizer.variables:
    print(v)
# 重みが共有されていることが分かります。順伝搬と逆伝搬を行います。
model.forward()
model.backward()
print(m1.W.g)  # type:ignore
print(m2.W.g)  # type:ignore
print(model.numerical_gradient(m1.W.variable))  # type:ignore
print(model.numerical_gradient(m2.W.variable))  # type:ignore
# 加算した勾配に等しいことが分かります。

# 次に、転置した重みを共有することを考えます。
net = [("input", L), ("affine", M), ("affine", L, "softmax_cross_entropy")]
trainer = sequential(net)
model = trainer.model
m1, m2, s = model.layers
w = np.random.randn(L, M)
m1.W.variable.data = w  # type:ignore
m2.W.variable.data = w.T.copy()  # type:ignore
# データを入力します。
model.set_data(x, t)
model.forward()
model.backward()
print(m1.W.g)  # type:ignore
print(model.numerical_gradient(m1.W.variable))  # type:ignore

print(m2.W.g)  # type:ignore
print(model.numerical_gradient(m2.W.variable))  # type:ignore
# 勾配を加算してみます。
print(m1.W.g + m2.W.g.T)  # type:ignore

# 重みを共有し、トレーナーをビルドします。このとき、`transpose`キーワードに`True`を指定します。
m2.W.share_variable(m1.W, transpose=True)  # type:ignore
trainer.build()
for v in trainer.optimizer.variables:
    print(v)
# 重みが共有されていることが分かります。順伝搬と逆伝搬を行います。
model.forward()
model.backward()
print(m1.W.g)  # type:ignore
print(m2.W.g)  # type:ignore
print(model.numerical_gradient(m1.W.variable))  # type:ignore
print(model.numerical_gradient(m2.W.variable))  # type:ignore
# 加算した勾配に等しいことが分かります。
