# ## Flatten

# Flattenレイヤは文字通り多次元配列を平坦化します。

from ivory.layers.core import Flatten

f = Flatten((2, 3))

for p in f.parameters:
    print(p)

# -
import numpy as np  # isort:skip

x = np.arange(2 * 2 * 3).reshape(2, 2, 3)
f.set_variables()
f.set_data(x)
print(f.x.d)
# -
f.forward()
print(f.y.d)
# 実装コードを確認しておきます。

# {{ from ivory.layers.core import Flatten }}
# ##Code <code>Flatten</code>クラス
# {{ Flatten # inspect }}
