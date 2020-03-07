import pickle
import sys

import pandas as pd

from ivory.common.context import np
from ivory.common.dataset import TimeDataset
from ivory.core.trainer import sequential
from ivory.utils.repository import repo_directory

np.context = "gpu"

sys.path.append(repo_directory("scratch2"))
from dataset import ptb  # isort:skip

corpus, _, _ = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
vocab_size = int(max(corpus) + 1)
x, t = corpus[:-1], corpus[1:]
data = TimeDataset((x, t), time_size=35, batch_size=20)
x, t = corpus_val[:-1], corpus_val[1:]
data_val = TimeDataset((x, t), time_size=35, batch_size=10)
print(data)
print(data_val)

# ハイパーパラメータの設定を行います。
wordvec_size = 650
hidden_size = 650
lr = 20.0
max_grad = 0.25
dropout = 0.5

# モデルを作成します。
net = [
    ("input", vocab_size),
    ("embedding", wordvec_size, "dropout"),
    ("lstm", hidden_size, "dropout"),
    ("lstm", hidden_size, "dropout"),
    ("affine", vocab_size, "softmax_cross_entropy"),
]
trainer = sequential(net, optimizer="sgd", metrics=["loss"])
trainer.optimizer.learning_rate = lr
trainer.max_grad = max_grad
model = trainer.model
model.init(std="xavier", dropout_ratio=dropout)
em, lstm1, lstm2, affine = model.layers[:-1:2]
affine.W.share_variable(em.W, transpose=True)  # type:ignore
trainer.build()


# 訓練を実施します。
trainer.fit(data)


def train_epoch(epoch):
    count = 0
    total_loss = 0
    for i, loss in trainer:
        total_loss += loss
        count += 1
        if i % 20 == 0:
            ppl = np.exp(total_loss / count)
            if i % 200 == 0:
                print(epoch + 1, i + 1, ppl)
            count = 0
            total_loss = 0
    return ppl


def validate():
    lstm1.reset_state()
    lstm2.reset_state()
    model.set_train(False)
    count = 0
    total_loss = 0
    for x, t in data_val:
        model.set_data(x, t)
        model.forward()
        total_loss += model.loss
        count += 1
    lstm1.reset_state()
    lstm2.reset_state()
    model.set_train(True)
    return np.exp(total_loss / count)


def save_weights():
    weights = [np.asnumpy(v.data) for v in model.weight_variables]
    with open("better_rnnlm.pkl", "wb") as f:
        pickle.dump(weights, f)


best_ppl = float("inf")
ppls = []
for epoch in range(40):
    ppl_train = train_epoch(epoch)
    ppl = validate()
    print("valid ppl: ", int(ppl))
    if best_ppl > ppl:
        best_ppl = ppl
        save_weights()
    else:
        lr /= 4.0
        trainer.optimizer.learning_rate = lr
    print("-" * 50)
    ppls.append([epoch + 1, ppl_train, ppl])

df = pd.DataFrame(ppls, columns=["epoch", "ppl_train", "ppl_val"])
df.to_csv("better_rnnlm_ppl.csv", index=False)
