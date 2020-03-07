# ## データセット

# Ivoryライブラリでは、学習するデータのセットをイテラブルとして実装します。おもちゃのデ
# ータを用意します。

import numpy as np

from ivory.common.dataset import Dataset

x_train = np.arange(0, 201).reshape(-1, 3)
t_train = np.arange(0, 201 // 3).reshape(-1, 1)
data = Dataset([x_train, t_train])
data
# `Dataset`クラスの`repr`は、内部状態を表示します。バッチサイズを変えてみます。
data.batch_size = 4
data
# データの長さ（＝バッチの個数）が{{len(data)}}に減りました。実際、
len(data)
# となります。`Dataset`クラスのインスタンスは通常のリストのようにインデクシングができます
# 。タプルが返されますので、各々の変数に代入するにはアンパックします。
x, t = data[0]
x
# -
t
# 取り出したデータの`shape`は`Dataset`の`shape`と一致します。
data.shape, x.shape, t.shape

# 先ほどの例では、データは先頭から取り出されていました。`random`属性を`True`にするとランダムにデ
# ータを取り出せます。
data.random = True
x, t = data[0]
x

# 通常は学習用のデータを「訓練データ」と「検証データ」のサブセットに分けます。（テストデ
# ータはまた別に用意するべきです。）`split`関数を使えばデータをサブセットに分割できます。
# 以下の例では3対1の大きさで2分割しています。
data.split((3, 1))
data
# `size`属性の要素数が変化したことが分かります。また、
len(data)
# となりました。どのサブセットからデータを取得するかは、`column`属性で指定します。
data.column = 1
data
# 取り出せるデータ数（`len`の値）が変わりました。実際に取り出してみます。
x, t = data[0]
x
# 多次元配列のように扱うことができます。第1要素がサブセット番号、第2要素がインデックスです。
x, t = data[0, 0]
print(t)
x, t = data[1, 0]
print(t)


# `column`が1のとき、元データの後半部分からデータを取得していることが分かります。サブセ
# ット間でデータを混ぜるには、`shuffle`関数を使います。
data.shuffle()
data.batch_size = 15
x, t = data[0, 0]
print(t.reshape(-1))
x, t = data[1, 0]
print(t.reshape(-1))
# `Dataset`はforループで使うことができます。
data.batch_size = 4
for k, (x, t) in enumerate(data):
    print(f"#{k}", t.reshape(-1))
# `epochs`を指定してイタレーション回数をコントロールできます。`epoch`属性、`index`属性
# 、`iteration`属性が`Dataset`のイタレーション状態を保持します。`epoch`属性は、エポックの
# 区切り以外では-1となります。また、`state`属性はこれらをまとめてタプル値を返します。
data.epochs = 2
for _ in data:
    print(data.epoch, data.index, data.iteration)
# `epochs`に-1を入力すると無限にループできます。
data.epochs = -1
for k, _ in enumerate(zip(range(1234), data)):
    pass
print(k, data.state)

# イテレータとして使うこともできます。分かりやすくするため、もう一度おもちゃのデータを作
# 成します。
x_train = np.arange(0, 201).reshape(-1, 3)
t_train = np.arange(0, 201 // 3).reshape(-1, 1)
data = Dataset([x_train, t_train])
data.split((2, 3, 4))
data.batch_size = 4
data
# -
it = iter(data)
x, t = next(it)
print(t.reshape(-1))
x, t = next(it)
print(t.reshape(-1))
data.column = 2
it = iter(data)
x, t = next(it)
print(t.reshape(-1))

# `Dataset`はスライス表記をサポートしています。
x, t = data[2:4]
print(len(x))
x, t = data[:]
print(len(x))
# データの一部だけを使いたいとき、`length`属性で大きさを制限できます。
data.length = 20
data
# スライス表記の結果を見てみます。
x, t = data[:]
len(x)
# `length`の値を-1にすると、元々の全データを使う状態に戻ります。
data.length = -1
print(data.size)
data
# -
x, t = data[:]
len(x)
