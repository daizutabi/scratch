# ## CUDAによる学習の高速化 {# scratch-context #}

# 「ゼロから作るDeep Learning」では第2巻でCuPyを導入しています。Ivoryライブラリでも同様
# にCUDAを使った学習ができる設計となっています。

# なお、Windows環境では、CuPyのインストール時にエラーが発生する場合があります。下記のよう
# にすると回避できます。（使用するCUDAのバージョンに合わせてパッケージ名を変更する必要が
# あります。）

# ~~~
# conda install fastrlock
# pip install cupy-cuda100
# ~~~

# 実際に使ってみます。
import cupy as cp

a = cp.arange(10)
print(type(a))
print(a.dtype)
print(a.device)


# Ivoryライブラリでは、`ivory.common.context`モジュールを使ってCPUとCUDAを切り替えます。
from ivory.common.context import np  # isort:skip

# コンテキスを確認します。
np.context
# NumPyと同じように使います。
a = np.arange(10, dtype="f")
print(type(a))
print(a.dtype)
# コンテキストを変更して、同じコードを試してみましょう。
np.context = "cuda"
a = np.arange(10, dtype="f")
print(type(a))
print(a.dtype)
# 今度はCuPyのアレイが得られました。このように同じコードで切り替えが可能です。参考までに
# 、`context`モジュールをソースを記載しておきます。

# ##File <code>ivory.common.context</code>モジュールファイル
# {%=/ivory/common/context.py%}

# {{ np.context = "cpu" }}
