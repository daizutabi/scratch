# # 2.3 カウントベースの手法

# [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/)で用
# 意されている関数をそのまま使います。

# # 2.3.1 Pythonによるコーパスの下準備
from ivory.common.util import preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(word_to_id)
print(id_to_word)

# # 2.3.4 共起行列
import pandas as pd  # isort:skip
from ivory.common.util import create_co_matrix  # isort:skip

C = create_co_matrix(corpus, 7)
v = list(id_to_word.values())
df = pd.DataFrame(C, index=v, columns=v)
df

# # 2.3.5 ベクトル間の類似度
from ivory.common.util import cos_similarity  # isort:skip

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
cos_similarity(c0, c1)

# # 2.3.6 類似単語のランキング表示
from ivory.common.util import most_similar  # isort:skip

most_similar("you", word_to_id, id_to_word, C, top=5)
