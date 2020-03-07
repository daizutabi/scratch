# # Feature Embedding (https://github.com/sony/nnabla-examples/tree/master/
# # mnist-collection#feature-embedding-siamesepy)

# MNISTのデータセットを使ったSiamese Neural Networkの例題を解いていく。まずは必要なモジ
# ュールのインポートを行う。
import os
from collections import namedtuple

import altair as alt
import nnabla as nn
import nnabla.functions as F
import nnabla.monitor as M
import nnabla.solvers as S
import numpy as np
import pandas as pd
from nnabla.ext_utils import get_extension_context

from ivory.utils.path import cache_dir
from ivory.utils.repository import import_module

# モジュールのインポートとコンテキストの設定。
I = import_module("nnabla-examples/mnist-collection/siamese")
context = get_extension_context("cudnn", device_id=0, type_config="float")
nn.set_default_context(context)
nn.context.get_current_context()

# 訓練用とテスト用にCNNを作成する。
net = namedtuple("net", ("image0", "image1", "label", "pred", "loss", "data"))


def create_net(test, batch_size=128, margin=1.0):  # Margin for contrastive loss.
    image0 = nn.Variable([batch_size, 1, 28, 28])
    image1 = nn.Variable([batch_size, 1, 28, 28])
    label = nn.Variable([batch_size])
    pred = I.mnist_lenet_siamese(image0, image1, test=test)
    loss = F.mean(I.contrastive_loss(pred, label, margin))
    data = I.siamese_data_iterator(batch_size, test, rng=np.random.RandomState(313))
    return net(image0, image1, label, pred, loss, data)


# 訓練関数を定義
def train(max_iter=5000, learning_rate=0.001, weight_decay=0):
    train = create_net(False)
    test = create_net(True)

    # ソルバーの作成
    solver = S.Adam(learning_rate)
    solver.set_parameters(nn.get_parameters())

    # モニタの作成
    path = cache_dir(os.path.join(I.name, "monitor"))
    monitor = M.Monitor(path)
    monitor_loss_train = M.MonitorSeries("training_loss", monitor, interval=100)
    monitor_time = M.MonitorTimeElapsed("time", monitor, interval=100)
    monitor_loss_val = M.MonitorSeries("val_loss", monitor, interval=100)

    # 訓練の実行
    for i in range(max_iter):
        if (i + 1) % 100 == 0:
            val_error = 0.0
            val_iter = 10
            for j in range(val_iter):
                test.image0.d, test.image1.d, test.label.d = test.data.next()
                test.loss.forward(clear_buffer=True)
                val_error += test.loss.d
            monitor_loss_val.add(i, val_error / val_iter)
        train.image0.d, train.image1.d, train.label.d = train.data.next()
        solver.zero_grad()
        train.loss.forward(clear_no_need_grad=True)
        train.loss.backward(clear_buffer=True)
        solver.weight_decay(weight_decay)
        solver.update()
        monitor_loss_train.add(i, train.loss.d.copy())
        monitor_time.add(i)

        nn.save_parameters(os.path.join(path, "params.h5"))
        return path


# 訓練を実行する。
path = train(max_iter=10000)


# 評価関数を定義
def evaluate(path):
    nn.load_parameters(os.path.join(path, "params.h5"))

    # Create embedded network
    batch_size = 500
    image = nn.Variable([batch_size, 1, 28, 28])
    feature = I.mnist_lenet_feature(image, test=True)

    # Process all images
    features = []
    labels = []

    # Prepare MNIST data iterator
    rng = np.random.RandomState(313)
    data = I.data_iterator_mnist(batch_size, train=False, shuffle=True, rng=rng)

    for i in range(10000 // batch_size):
        image_data, label_data = data.next()
        image.d = image_data / 255.0
        feature.forward(clear_buffer=True)
        features.append(feature.d.copy())
        labels.append(label_data.copy())
    features = np.vstack(features)
    labels = np.vstack(labels)
    df = pd.DataFrame(features, columns=["x", "y"])
    df["label"] = labels

    return df


# 可視化してみる。
df = evaluate(path)
df.head()


# -
alt.Chart(df.sample(2000)).mark_point().encode(
    x="x", y="y", color="label:N"
).properties(width=250, height=250)
