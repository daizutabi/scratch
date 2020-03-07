# # Deep Convolutional GAN (https://github.com/sony/nnabla-examples/tree/master/
# # mnist-collection#deep-convolutional-gan-dcganpy)

# Reference: ["Unsupervised Representation Learning with Deep Convolutional Generative
# Adversarial Networks"](https://arxiv.org/abs/1507.00677).

# ## サンプルコードの実行

# MNISTのデータセットを使ったGenerative Adversarial Networksの例題を解いていく。まずは必
# 要なモジュールのインポートを行う。
import inspect
import os

import altair as alt
import nnabla as nn
import nnabla.functions as F
import nnabla.monitor as M
import nnabla.solvers as S
import numpy as np
from nnabla.ext_utils import get_extension_context

from ivory.utils.path import cache_dir
from ivory.utils.repository import import_module
from ivory.utils.nnabla.monitor import read_monitor

# `import_module`はGitレポジトリからモジュールをインポートする。
I = import_module("nnabla-examples/mnist-collection/dcgan")
I.__file__

# これで、 mnist-collection/dcgan.pyの内部にアクセスできるようになった。 今回の例ではハイ
# パーパラメータが設定されているのでそれに倣う。

source = inspect.getsource(I)
print(source[source.index("if __name__") :])

max_iter = 20000
learning_rate = 0.0002
batch_size = 64
weight_decay = 0.0001


# コンテキストを設定する。
context = get_extension_context("cudnn", device_id=0, type_config="float")
nn.set_default_context(context)
nn.get_current_context()

# Fakeパスの設定
z = nn.Variable([batch_size, 100, 1, 1])
fake = I.generator(z)
fake.persistent = True  # Not to clear at backward
pred_fake = I.discriminator(fake)
loss_gen = F.mean(F.sigmoid_cross_entropy(pred_fake, F.constant(1, pred_fake.shape)))
fake_dis = fake.get_unlinked_variable(need_grad=True)
fake_dis.need_grad = True  # TODO: Workaround until v1.0.2
pred_fake_dis = I.discriminator(fake_dis)
loss_dis = F.mean(
    F.sigmoid_cross_entropy(pred_fake_dis, F.constant(0, pred_fake_dis.shape))
)

# Realパスの設定
x = nn.Variable([batch_size, 1, 28, 28])
pred_real = I.discriminator(x)
loss_dis += F.mean(F.sigmoid_cross_entropy(pred_real, F.constant(1, pred_real.shape)))

# ソルバーの作成
solver_gen = S.Adam(learning_rate, beta1=0.5)
solver_dis = S.Adam(learning_rate, beta1=0.5)
with nn.parameter_scope("gen"):
    solver_gen.set_parameters(nn.get_parameters())
with nn.parameter_scope("dis"):
    solver_dis.set_parameters(nn.get_parameters())

# パラメータスコープの使い方を見ておく。
print(len(nn.get_parameters()))
with nn.parameter_scope("gen"):
    print(len(nn.get_parameters()))
# パラメータスコープ内では、`get_parameters()`で取得できるパラメータがフィルタリングされ
# る。


# モニターの設定
path = cache_dir(os.path.join(I.name, "monitor"))
monitor = M.Monitor(path)
monitor_loss_gen = M.MonitorSeries("generator_loss", monitor, interval=100)
monitor_loss_dis = M.MonitorSeries("discriminator_loss", monitor, interval=100)
monitor_time = M.MonitorTimeElapsed("time", monitor, interval=100)
monitor_fake = M.MonitorImageTile(
    "Fake images", monitor, normalize_method=lambda x: (x + 1) / 2.0
)


# パラメータ保存関数の定義
def save_parameters(i):
    with nn.parameter_scope("gen"):
        nn.save_parameters(os.path.join(path, "generator_param_%06d.h5" % i))
    with nn.parameter_scope("dis"):
        nn.save_parameters(os.path.join(path, "discriminator_param_%06d.h5" % i))


# 訓練の実行
def train(max_iter):
    data = I.data_iterator_mnist(batch_size, True)
    for i in range(max_iter):
        if i % 1000 == 0:
            save_parameters(i)
        # Training forward
        image, _ = data.next()
        x.d = image / 255.0 - 0.5  # [0, 255] to [-1, 1]
        z.d = np.random.randn(*z.shape)

        # Generator update.
        solver_gen.zero_grad()
        loss_gen.forward(clear_no_need_grad=True)
        loss_gen.backward(clear_buffer=True)
        solver_gen.weight_decay(weight_decay)
        solver_gen.update()
        monitor_fake.add(i, fake)
        monitor_loss_gen.add(i, loss_gen.d.copy())

        # Discriminator update.
        solver_dis.zero_grad()
        loss_dis.forward(clear_no_need_grad=True)
        loss_dis.backward(clear_buffer=True)
        solver_dis.weight_decay(weight_decay)
        solver_dis.update()
        monitor_loss_dis.add(i, loss_dis.d.copy())
        monitor_time.add(i)

    save_parameters(i)


train(max_iter=20000)

# 可視化
df = read_monitor(path, melt="loss")
df.head()

# -
alt.Chart(df).mark_line(point=True).encode(x="step", y="loss", color="type").properties(
    width=200, height=150
)
