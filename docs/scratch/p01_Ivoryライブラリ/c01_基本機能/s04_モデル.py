# ## モデル

# 一連のレイヤよりなるモデルは`model`モジュールの`sequential`関数で構築することができます。
# このメソッドはレイヤクラス名をスネークケースで記述した「レイヤ表現」を引数に取り
# 、`Model`クラスのインスタンスを返します。最終レイヤは損失関数を出力するレイヤである必要
# があります。各レイヤのパラメータには変数が自動で割り当てられます。

from ivory.core.model import sequential

net = [("input", 5), ("affine", 10, "relu"), ("affine", 4, "softmax_cross_entropy")]
model = sequential(net)
model.layers

# ここで、最初の`("input", 5)`が、5次元の入力を示すプレースホルダです。上記のよう
# に`Model`インスタンスは`layers`属性を持ち、所属するレイヤを返します。各レイヤにはクラス
# ごとに連番で名前が付けられています。

# `sequential`関数は、同じパターンの繰り返しを簡潔に書く方法が用意されています。

net = [
    ("input", 784),
    (3, "affine", 100, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
model_complex = sequential(net)
model_complex.layers

# このように、レイヤ表現の最初の要素が整数の場合、その数だけ残りの要素を繰り返します。

# `layers`属性のほかにも、損失に影響を与えるパラメータを取得できます。
model.inputs
# -
model.outputs
# -
model.weights
# -
model.losses
# 期待通りの結果が得られました。パラメータだけでなく、変数の取得もできます。
model.input_variables
# -
model.output_variables
# -
model.weight_variables
# `inputs_variables`と`output_variables`を比べると一部重複していることが分かります。これ
# は、あるレイヤの出力が次段のレイヤの入力となるためです。外部から入力する変数、もしくは
# 最終的な結果を出力する変数は、以下のように取得できます。
model.data_input_variables
# -
model.data_output_variables
# `data_input_variables`にデータを与えて、ニューラルネットワークを評価し、出力
# を`data_output_variables`から取り出します。損失の評価は、`loss_variables`を通じて行いま
# す。
model.loss_variables
# 学習は、これらの`Variable`インスタンスを介して行われます。
