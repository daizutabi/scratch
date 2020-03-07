# # Semi-Supervised Learning of Classification (https://github.com/sony/
# # nnabla-examples/tree/master/
# # mnist-collection#semi-supervised-learning-of-classification-vatpy)

# Reference: ["Distributional Smoothing with Virtual Adversarial
# Training"](https://arxiv.org/abs/1507.00677)

# MNISTのデータセットを使ったVirtual Adversarial Trainingの例題を解いていく。まずは必要
# なモジュールのインポートを行う。
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
from ivory.utils.utils import set_args
from ivory.utils.nnabla.monitor import read_monitor

# 例題モジュールのインポートとコンテキストの設定。
I = import_module("nnabla-examples/mnist-collection/vat")
context = get_extension_context("cudnn", device_id=0, type_config="float")
nn.set_default_context(context)
nn.context.get_current_context()


# パラメータの設定。
set_args()
args = I.get_args()

for attr in dir(args):
    if not attr.startswith("_"):
        print(attr + ":", getattr(args, attr))


# 訓練関数の定義
def train(max_iter=24000):
    shape_x = (1, 28, 28)
    n_h = args.n_units
    n_y = args.n_class

    # Load MNIST Dataset
    from mnist_data import load_mnist, data_iterator_mnist

    images, labels = load_mnist(train=True)
    rng = np.random.RandomState(706)
    inds = rng.permutation(len(images))

    def feed_labeled(i):
        j = inds[i]
        return images[j], labels[j]

    def feed_unlabeled(i):
        j = inds[i]
        return images[j], labels[j]

    di_l = I.data_iterator_simple(
        feed_labeled,
        args.n_labeled,
        args.batchsize_l,
        shuffle=True,
        rng=rng,
        with_file_cache=False,
    )
    di_u = I.data_iterator_simple(
        feed_unlabeled,
        args.n_train,
        args.batchsize_u,
        shuffle=True,
        rng=rng,
        with_file_cache=False,
    )
    di_v = data_iterator_mnist(args.batchsize_v, train=False)

    # Create networks
    # feed-forward-net building function
    def forward(x, test=False):
        return I.mlp_net(x, n_h, n_y, test)

    # Net for learning labeled data
    xl = nn.Variable((args.batchsize_l,) + shape_x, need_grad=False)
    yl = forward(xl, test=False)
    tl = nn.Variable((args.batchsize_l, 1), need_grad=False)
    loss_l = F.mean(F.softmax_cross_entropy(yl, tl))

    # Net for learning unlabeled data
    xu = nn.Variable((args.batchsize_u,) + shape_x, need_grad=False)
    yu = forward(xu, test=False)
    y1 = yu.get_unlinked_variable()
    y1.need_grad = False

    noise = nn.Variable((args.batchsize_u,) + shape_x, need_grad=True)
    r = noise / (F.sum(noise ** 2, [1, 2, 3], keepdims=True)) ** 0.5
    r.persistent = True
    y2 = forward(xu + args.xi_for_vat * r, test=False)
    y3 = forward(xu + args.eps_for_vat * r, test=False)
    loss_k = F.mean(I.distance(y1, y2))
    loss_u = F.mean(I.distance(y1, y3))

    # Net for evaluating validation data
    xv = nn.Variable((args.batchsize_v,) + shape_x, need_grad=False)
    hv = forward(xv, test=True)
    tv = nn.Variable((args.batchsize_v, 1), need_grad=False)
    err = F.mean(F.top_n_error(hv, tv, n=1))

    # Create solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitor training and validation stats.
    path = cache_dir(os.path.join(I.name, "monitor"))
    monitor = M.Monitor(path)
    monitor_verr = M.MonitorSeries("val_error", monitor, interval=240)
    monitor_time = M.MonitorTimeElapsed("time", monitor, interval=240)

    # Training Loop.
    for i in range(max_iter):

        # Validation Test
        if i % args.val_interval == 0:
            valid_error = I.calc_validation_error(di_v, xv, tv, err, args.val_iter)
            monitor_verr.add(i, valid_error)

        # forward, backward and update
        xl.d, tl.d = di_l.next()
        xl.d = xl.d / 255
        solver.zero_grad()
        loss_l.forward(clear_no_need_grad=True)
        loss_l.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        # Calculate y without noise, only once.
        xu.d, _ = di_u.next()
        xu.d = xu.d / 255
        yu.forward(clear_buffer=True)

        # Do power method iteration
        noise.d = np.random.normal(size=xu.shape).astype(np.float32)
        for k in range(args.n_iter_for_power_method):
            r.grad.zero()
            loss_k.forward(clear_no_need_grad=True)
            loss_k.backward(clear_buffer=True)
            noise.data.copy_from(r.grad)

        # forward, backward and update
        solver.zero_grad()
        loss_u.forward(clear_no_need_grad=True)
        loss_u.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        if i % args.iter_per_epoch == 0:
            solver.set_learning_rate(solver.learning_rate() * args.learning_rate_decay)
        monitor_time.add(i)

    # Evaluate the final model by the error rate with validation dataset
    valid_error = I.calc_validation_error(di_v, xv, tv, err, args.val_iter)
    monitor_verr.add(i, valid_error)
    monitor_time.add(i)

    return path


# 訓練の実行
path = train(max_iter=24000)

# 可視化
df = read_monitor(path, melt="error")
df.head()

# -
alt.Chart(df).mark_line(point=True).encode(
    x="step", y=alt.Y("error", scale={"type": "log"}), color="type"
).properties(width=200, height=150)
