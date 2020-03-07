# # 7.1 言語モデルを使った文章生成

# 高速化のためにGPUを使います。
from ivory.common.context import np  # isort:skip
np.context = 'gpu'

# PTBデータセットを読み出します。
from ivory.utils.repository import import_module  # isort:skip

ptb = import_module("scratch2/dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data("train")

# モデルを作成します。
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 10000),
    ("embedding", 650),
    ("lstm", 650),
    ("lstm", 650),
    ("affine", 10000, "softmax_cross_entropy"),
]
model = sequential(net)

# 重みの共有をします。
em = model.layers[0]
affine = model.layers[-2]
affine.W.share_variable(em.W, transpose=True)  # type:ignore
model.build()

# 学習済みの重みを読み出します。
import os  # isort:skip
import pickle  # isort:skip
import ivory  # isort:skip

directory = os.path.dirname(ivory.__file__)
directory = os.path.join(directory, "../docs/script")
with open(os.path.join(directory, 'better_rnnlm.pkl'), 'rb') as f:
    weights = pickle.load(f)

for v, weight in zip(model.weight_variables, weights):
    v.data = np.asarray(weight)

# start文字とskip文字を設定します。
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]


# Softmax関数を定義します。
def softmax(x):
    y = np.exp(x - x.max())
    return y / y.sum()


# 文章ジェネレータを定義します。
def generate(word_id, skip_ids=None):
    score = model.layers[-1].x
    yield id_to_word[word_id]
    while True:
        x = np.array(word_id).reshape(1, 1)
        model.set_data(x)
        model.forward(predict=True)
        p = softmax(score.d.flatten())
        sampled = np.random.choice(len(p), size=1, p=p)
        if skip_ids is None or sampled not in skip_ids:
            word_id = int(sampled)
            yield id_to_word[word_id]


# 文章を生成します。
sentences = []
for _ in range(5):
    gen = generate(start_id, skip_ids)
    words = [word for word, _ in zip(gen, range(100))]
    sentences.append(" ".join(words))

# * {{ sentences[0] }}

# * {{ sentences[1] }}

# * {{ sentences[2] }}

# * {{ sentences[3] }}

# * {{ sentences[4] }}
