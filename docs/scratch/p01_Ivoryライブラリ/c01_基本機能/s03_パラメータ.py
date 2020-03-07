# ## パラメータ

# Affineレイヤを作成します。

from ivory.layers.affine import Affine

affine = Affine((2, 3))
affine

# 重み`weights`を覗いてみます。

affine.W

# 重みは2次元配列なので、`Variable`インスタンスでよさそうです。しかしながら、実際は本節で
# 説明する`Parameter`インスタンスになっています。

from ivory.core.parameter import Parameter  # isort:skip

isinstance(affine.W, Parameter)

# 実際はそのサブクラス`Weight`のインスタンスです。

type(affine.W)

# 他も見てみます。

print(type(affine.x))
print(type(affine.b))
print(type(affine.weight_decay))
print(type(affine.y))

# 属性を確認します。
vars(affine.W)
# `layer`属性によって自分がどのレイヤに属しているかが分かります。

# `Parameter`には実データを保持する`Variable`インスタンスを生成して自分に割り当て
# る`set_variable`メソッドが用意されています。

v = affine.W.set_variable()
v

# 値を確認してみます。
print(v.data, v.data.dtype)
# 乱数で初期化されています。

# `Parameter`インスタンスからも変数のデータと勾配にアクセスできます。この場合には`d`プロ
# パティと`g`プロパティを使います。

print(affine.W.d)
print(affine.W.g)

# 勾配はまだ計算されていません。

# 値を代入できるでしょうか？

affine.W.d = 0

# 重みに関してはデータの読み出しは可能ですが、書き込みは不可能です（当然、`W.d[0, 0] =
# 0`といった書き換えは可能ですが）。なぜならレイヤ自身が重みの値を直接操作することは必要
# ないからです。出力に関してはどうでしょうか。適当な値を代入してみます。（以下では簡単の
# ために、スカラー値を代入しますが、実際にはバッチ数×データ次元の2次元配列になります。）

affine.y.set_variable()
affine.y.d = 1
affine.y.d

# レイヤは、所望の計算を行って値を出力する必要があるので、変数に値を代入できるのは当然で
# す。もう一回代入してみます。

affine.y.d = 2
affine.y.d

# 通常では2となると思われますが、加算されて3になりました。これは、パラメータを介した変数
# への値の代入が通常とは違う論理で行われているからです。同じ変数に複数回代入があるという
# ことは、その変数に複数のレイヤから値が出力されたことを意味します。それぞれの出力を加算
# することは暗黙的にSumノードを想定しています。このようにしているは、逆伝搬において分岐し
# たノードからの勾配が加算されることに対称な動作とするためです。実際、

affine.x.set_variable()
affine.x.g = 1
affine.x.g = 1
affine.x.g

# となります。ところで、
affine.x.d = 1
affine.y.g = 1

# となり、入力のデータおよび出力の勾配には値の代入ができません。順伝搬、逆伝搬の動作を考
# えれば妥当な仕様です。以下にパラメータの種類と挙動を示します。

# ~~~markdown
# |クラス名  | データ  |勾配  |
# |---|---|---|
# |`Input` | 代入不可 | 加算代入 |
# |`Output` | 加算代入 | 代入不可 |
# |`Weight` | 代入不可 | 加算代入 |
# |`State` | 通常代入 | なし |
# |`Loss` | 加算代入 | なし |
# ~~~

# 次に、Affineレイヤに別のAffineレイヤを接続することを考えます。1つ目の出力と2つ目の入力
# に同じ変数を割り当てるために、`Variable.add_parameter`メソッドを使います。

a = Affine((2, 3))
b = Affine((3, 2))
v = a.y.set_variable()
print(v)
v.add_parameter(b.x)
print(v)

# 上記のように、`Variable`の`repr`関数は、接続されている`Parameter`のリストを表示します。
# 動作を確認します。
a.y.d = 1
print(b.x.d)
b.x.g = 10
print(a.y.g)
# このように値が伝搬することが確認できます。

# データおよび勾配が加算される理由は、以下のような並列構造で明らかになります。
a = Affine((1, 1), name="A")
b = Affine((1, 1), name="B")
c = Affine((1, 1), name="C")
x = a.x.set_variable()
x.add_parameter(b.x)
x.add_parameter(c.x)
y = a.y.set_variable()
y.add_parameter(b.y)
y.add_parameter(c.y)
print(x)
print(y)

# それぞれのレイヤが別の値を出力したとします。
y.data = 0
a.y.d = 1
b.y.d = 2
c.y.d = 3
print(y.data)
# 変数の値は加算されたものとなります。逆伝搬も同じです。
x.grad = 0
a.x.g = 10
b.x.g = 20
c.x.g = 30
print(x.grad)

# 損失関数の例を見てみます。
from ivory.layers.loss import SoftmaxCrossEntropy  # isort:skip

s = SoftmaxCrossEntropy((3,))
print(type(s.y))
print(type(s.loss))
# 損失パラメータ`loss`は特別な`Loss`クラスのインスタンスです。ニューラルネットワークの学
# 習では、損失に対して各種の操作を行うため、Ivoryライブラリでは特別のクラスを与えています
# 。

# なお、実験用に、レイヤを作成した後に、全てのパラメータに変数を自動で割り当てるメソッ
# ド`Layer.set_variables`も用意されています。
s.set_variables()
# `set_data`メソッドを使えば、入力データをセットできます。
s.set_data([1, 2, 3], 2)
print(s.x.d)
print(s.t.d)
