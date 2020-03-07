# # 5.7 誤差逆伝搬法の実装

# [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) 5章7節
# の内容に合わせて、Ivoryライブラリを使って実際に学習を行ってみます。

# ### 訓練データの取得

# MNIST手書き文字の訓練データは、Datasetとして用意されています。
from ivory.datasets.mnist import load_dataset

data_train, data_test = load_dataset()
data_train.random = True
data_train

# # 5.7.2 誤差逆伝搬法に対応したニューラルネットワークの実装
from ivory.core.model import sequential  # isort:skip

net = [
    ("input", 784),
    ("affine", 50, "relu"),
    ("affine", 10, "softmax_cross_entropy"),
]
model = sequential(net)
model

# ### 誤差逆伝搬法を使った学習

# 最後に学習の実装です。「ゼロから作るDeep Learning」5章7節のコードを参考にします。

iters_num = 10000
train_size = data_train.data[0].shape[0]
data_train.batch_size = 100
iter_per_epoch = train_size // data_train.batch_size
data_train.epochs = -1
learning_rate = 0.1

for i, (x, t) in zip(range(iters_num), data_train):
    model.set_data(x, t)
    model.forward()
    model.backward()

    for variable in model.weight_variables:
        variable.data -= learning_rate * variable.grad  # type:ignore

    if i % iter_per_epoch == 0:
        x, t = data_train[:]
        model.set_data(x, t)
        model.forward()
        train_acc = model.accuracy
        x, t = data_test[:]
        model.set_data(x, t)
        model.forward()
        test_acc = model.accuracy
        print(f"{train_acc:.3f}", f"{test_acc:.3f}")


# 「ゼロから作るDeep Learning」レポジトリのコードを実行してみます。
from ivory.utils.repository import run

run("scratch/ch05/train_neuralnet.py")


# 実行速度の違いは、Ivoryライブラリでのパラメータのビット精度32ビットであるためです。
