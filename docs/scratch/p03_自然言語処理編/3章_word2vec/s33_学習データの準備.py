# # 3.3 学習データの準備

# [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/)で用
# 意されている関数をそのまま使います。

# ### コンテキストとターゲット
from ivory.common.util import create_contexts_target, preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)

# ### one-hot表現への変換

# IvoryライブラリではSoftmaxCrossEntropyレイヤはラベル表現を入力にとるので、コンテキスト
# のみone-hot表現へ変換します。
from ivory.common.util import convert_one_hot  # isort:skip

vocab_size = len(word_to_id)
contexts = convert_one_hot(contexts, vocab_size)
print(contexts.shape)
print(target.shape)
