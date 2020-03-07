# # Variational Auto-encoder (https://github.com/sony/nnabla-examples/tree/master/
# # mnist-collection#variational-auto-encoder-vaepy)

# Reference: ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114)

# MNISTのデータセットを使ったVariational Auto-encoderの例題を解いていく。まずは必要なモ
# ジュールのインポートを行う。
import os

import altair as alt
import nnabla as nn
import nnabla.monitor as M
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context

from ivory.utils.path import cache_dir
from ivory.utils.repository import import_module
from ivory.utils.nnabla.monitor import read_monitor

# 例題モジュールのインポートとコンテキストの設定。
I = import_module("nnabla-examples/mnist-collection/vae")
context = get_extension_context("cudnn", device_id=0, type_config="float")
nn.set_default_context(context)
nn.context.get_current_context()


# デフォルトパラメータの設定。
learning_rate = 3e-4
batch_size = 100
weight_decay = 0


# 訓練関数の定義。
def train(max_iter=60000):
    # Initialize data provider
    di_l = I.data_iterator_mnist(batch_size, True)
    di_t = I.data_iterator_mnist(batch_size, False)

    # Network
    shape_x = (1, 28, 28)
    shape_z = (50,)
    x = nn.Variable((batch_size,) + shape_x)
    loss_l = I.vae(x, shape_z, test=False)
    loss_t = I.vae(x, shape_z, test=True)

    # Create solver
    solver = S.Adam(learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitors for training and validation
    path = cache_dir(os.path.join(I.name, "monitor"))
    monitor = M.Monitor(path)
    monitor_train_loss = M.MonitorSeries("train_loss", monitor, interval=600)
    monitor_val_loss = M.MonitorSeries("val_loss", monitor, interval=600)
    monitor_time = M.MonitorTimeElapsed("time", monitor, interval=600)

    # Training Loop.
    for i in range(max_iter):

        # Initialize gradients
        solver.zero_grad()

        # Forward, backward and update
        x.d, _ = di_l.next()
        loss_l.forward(clear_no_need_grad=True)
        loss_l.backward(clear_buffer=True)
        solver.weight_decay(weight_decay)
        solver.update()

        # Forward for test
        x.d, _ = di_t.next()
        loss_t.forward(clear_no_need_grad=True)

        # Monitor for logging
        monitor_train_loss.add(i, loss_l.d.copy())
        monitor_val_loss.add(i, loss_t.d.copy())
        monitor_time.add(i)

    return path


# 訓練の実行
path = train(max_iter=60000)

# 可視化
df = read_monitor(path, melt="loss")
df.head()

# -
alt.Chart(df).mark_line(point=True).encode(x="step", y="loss", color="type").properties(
    width=200, height=150
)
