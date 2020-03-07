# # 2.4 カウントベースの手法の改善

# [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/)で用
# 意されている関数をそのまま使います。

# # 2.4.1 相互情報量
import pandas as pd

from ivory.common.util import create_co_matrix, ppmi, preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
v = list(id_to_word.values())
df = pd.DataFrame(W, index=v, columns=v)
df

# # 2.4.3 SVDによる次元削減
import numpy as np  # isort:skip

U, S, V = np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])

# プロットしてみます。 {{ import matplotlib.pyplot as plt }}
import matplotlib.pyplot as plt  # isort:skip

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

# # 2.4.4 PTBデータセット
from ivory.utils.repository import import_module  # isort:skip

ptb = import_module("scratch2/dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data("train")
len(corpus)

# # 2.4.5 PTBデータセットでの評価
from sklearn.utils.extmath import randomized_svd  # isort:skip
from ivory.common.util import most_similar  # isort:skip

window_size = 2
wordvec_size = 100
vocab_size = len(word_to_id)

print("counting  co-occurrence ...")
C = create_co_matrix(corpus, vocab_size, window_size)
print("calculating PPMI ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
