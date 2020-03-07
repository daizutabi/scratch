# ## Negative Sampling

# [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/)で4章
# で導入されるNegative Samplingを実装します。元になる関数は、「ゼロから作るDeep Learning
# ❷」をほぼ踏襲し、Ivoryライブラリの`Dataset`のサブクラス`ContextDataset`として使用でき
# るようにします。

# 実験用のコーパスを準備します。
corpus = [0, 1, 2, 3, 4, 1, 2, 3, 2]

# コンテキスト用データセットを作成します。
from ivory.common.dataset import ContextDataset  # isort:skip

data = ContextDataset(corpus)
data.batch_size = 2
data
# ウィンドウサイズはデフォルトで1です。
data.window_size
# 通常の`Dataset`と同様に動作します。コンテキストとターゲットを返します。
data[0]
# ウィンドウサイズは、後から変更することもできます。
data.set_window_size(2)
data
# データの長さが変化しました。データを取得します。
data[0]
# Nagative Samplingを行うためには、`negative_sampling_size`を指定します。
data.negative_sample_size = 2
data[0]
# 第3要素以降が負例になります。デフォルトでは、ターゲットとの重複を許しません。
data.negative_sample_size = 4
data[0]
# `replace`属性を`True`にすることで、重複を許します。すなわち、ターゲットが負例として現
# れることを（速度を優先した結果として）許容します。
data.replace = True
data.negative_sample_size = 10
data[0]
