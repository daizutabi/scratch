# ## レイヤ

# ネットワークを構成するレイヤは、`Layer`クラスのインスタンスが表現します。Affineレイヤを
# 見てみましょう。

from ivory.core.layer import Layer
from ivory.layers.affine import Affine

assert issubclass(Affine, Layer)
affine = Affine((2, 3))
affine

# レイヤインスタンスは、入力`inputs`、重み`weights`、状態`states`、そして、出
# 力`outputs`の属性を持ちます。これらは複数持つことができます。

for attr in ["inputs", "weights", "states", "outputs"]:
    for var in getattr(affine, attr):
        print(var)

# 属性には適切な名前がついているのが確認できます。実際に、レイヤインスタンスから直接名前
# でアクセスできます。

affine.W

# 上記で`Affine({{affine.shape}})`としたので、重み`W`とバイアス`b`がそれにふさわしい形状
# で作成されています。

print(affine.W.shape, affine.b.shape)

# 一方で、入力`x`と出力`y`の形状は、

print(affine.x.shape, affine.y.shape)

# となり、バッチサイズは含まれていません。

# ニューラルネットワークの学習では、損失関数の値を最小化するように重みを調整します。損失
# 関数を含むレイヤは`LossLayer`という特別なクラスで実装されています。例を見てみます。
from ivory.layers.loss import LossLayer, SoftmaxCrossEntropy  # isort:skip

s = SoftmaxCrossEntropy((3,))
print(isinstance(s, LossLayer))
print(s)

# 上記は、ソフトマックス関数と交差エントロピー誤差がセットになったレイヤです。形状
# が(3,)となっているため、3種類のクラスを分類します。入力は前段レイヤからの入力`x`とター
# ゲット値`t`を持ちます。`t`はone-hot表現ではなく、ラベル表現をとるため、スカラー値になり
# ます。

s.inputs

# 「出力」は2つあります。一つは、ソフトマックス関数の出力です。

s.outputs
# これは各クラスの確率を与えるため、入力と同じ形状をしています。もう一つは損失である交差
# エントロピー誤差であり、こちらはスカラー値です。`loss`属性でアクセスできます。

s.loss
