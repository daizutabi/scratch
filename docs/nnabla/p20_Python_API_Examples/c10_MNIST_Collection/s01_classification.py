# #!

# # Classification task (https://github.com/sony/nnabla-examples/tree/master/
# # mnist-collection#classification-task-classificationpy-and-classification_bnnpy)

# MNISTの分類の例題を解いていく。まずは必要なモジュールのインポートを行う。
import os
from collections import namedtuple

import altair as alt
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import numpy as np
import pandas as pd
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from numpy.random import RandomState

# Ivoryパッケージから便利な関数をインポート。
from ivory.utils.path import cache_dir, cache_file
from ivory.utils.repository import import_module
from ivory.utils.utils import set_args
from ivory.utils.nnabla.monitor import read_monitor

# `import_module`はGitレポジトリからモジュールをインポートする。
cl = import_module("nnabla-examples/mnist-collection/classification")
cl_bnn = import_module("nnabla-examples/mnist-collection/classification_bnn")

print(cl.__file__)
print(cl_bnn.__file__)

# これで、 mnist-collection/classification[_bnn].pyの内部にアクセスできるようになった。
# `set_args`関数は、`sys.argv`を書き換えて、あたかもコンソールから実行しているかのように
# 装うためにある。コンテキストをcudnnに設定し、保存ディレクトリを指定する。

save_path = cache_file(cl.name, "tmp.monitor")
set_args(f"--context cudnn -m {save_path} -o {save_path}")
args = cl.get_args()

# 変数`args`を見てみる。
print("Type:", type(args))
for attr in dir(args):
    if not attr.startswith("_"):
        print(attr + ":", getattr(args, attr))


# コンテキストを設定する。
print(args.context, args.device_id, args.type_config)
context = get_extension_context(
    args.context, device_id=args.device_id, type_config=args.type_config
)
nn.set_default_context(context)
nn.get_current_context()


# 用意されているネットを確認する。
nets = [attr for m in [cl, cl_bnn] for attr in dir(m) if attr.endswith("prediction")]
nets = [net.replace("mnist_", "").replace("_prediction", "") for net in nets]
nets


# ネット関数を返す関数を定義する。
def get_net_prediction(net_name: str):
    module = cl_bnn if net_name.startswith("binary") else cl
    return getattr(module, f"mnist_{net_name}_prediction")


# モデルを作成する関数を定義する。
def create_model(net_name: str, batch_size=128, learning_rate=0.001):
    nn.clear_parameters()
    net = namedtuple("net", ("image", "label", "pred", "loss"))
    model = namedtuple("model", ("name", "train", "val", "solver"))
    mnist_cnn_prediction = get_net_prediction(net_name)

    def create_net(test, persistent):
        image = nn.Variable([batch_size, 1, 28, 28])
        label = nn.Variable([batch_size, 1])
        norm = 255 if net_name.startswith("binary") else 1
        pred = mnist_cnn_prediction(image / norm, test=test)
        if persistent:
            pred.persistent = True
        loss = F.mean(F.softmax_cross_entropy(pred, label))
        return net(image, label, pred, loss)

    net_train = create_net(test=False, persistent=True)
    net_val = create_net(test=True, persistent=False)
    solver = S.Adam(learning_rate)
    solver.set_parameters(nn.get_parameters())

    return model(net_name, net_train, net_val, solver)


# デフォルトのネット(args.net='{{args.net}}')でモデルを作成する。
model = create_model(args.net)
print(model.train.image)
print(model.train.label)
print(model.train.pred)
print(model.train.loss)
print(model.solver)


# `parameters`の内容を確認する。
nn.get_parameters()

# -
model.solver.get_parameters()

# 同じオブジェクトではない。順番も入れ替わっている。


# モニターを作成する関数を定義する。
def get_monitor_path(net_name: str) -> str:
    return cache_dir(os.path.join(cl.name, net_name))


def create_monitor(net_name: str, interval=100):
    value = namedtuple("value", ("loss", "error"))
    monitor = namedtuple("monitor", ("path", "time", "train", "val"))

    path = get_monitor_path(net_name)
    kwargs = dict(monitor=Monitor(path), interval=interval)
    time = MonitorTimeElapsed("time", **kwargs)

    kwargs["verbose"] = False
    loss = MonitorSeries("train_loss", **kwargs)
    error = MonitorSeries("train_error", **kwargs)
    value_train = value(loss, error)

    loss = MonitorSeries("val_loss", **kwargs)
    error = MonitorSeries("val_error", **kwargs)
    value_val = value(loss, error)

    return monitor(path, time, value_train, value_val)


# データイタレータを作成する関数を定義する。バッチサイズごとにデータをイールドする。
def create_data_iterator(batch_size=128):
    iterator = namedtuple("iterator", ("train", "val"))

    train = cl.data_iterator_mnist(batch_size, train=True, rng=RandomState(1223))
    val = cl.data_iterator_mnist(batch_size, train=False)
    return iterator(train, val)


# エラーを計算する関数を定義しておく。
def calc_error(net, data, iteration=10):
    def error():
        net.image.d, net.label.d = data.next()
        net.pred.forward(clear_buffer=True)
        net.pred.data.cast(np.float32, context)
        return cl.categorical_error(net.pred.d, net.label.d)

    return sum(error() for _ in range(iteration)) / iteration


# 訓練を実行する関数を定義する。
def train(model, max_iter=10000, weight_decay=0, val_interval=100, save_interval=1000):
    data = create_data_iterator()
    monitor = create_monitor(model.name)
    for i in range(max_iter):
        if (i + 1) % val_interval == 0:
            error = calc_error(model.val, data.val)
            monitor.val.error.add(i, error)
        if (i + 1) % save_interval == 0:
            nn.save_parameters(os.path.join(monitor.path, "params_%06d.h5" % i))

        model.train.image.d, model.train.label.d = data.train.next()
        model.solver.zero_grad()
        model.train.loss.forward(clear_no_need_grad=True)
        model.train.loss.backward(clear_buffer=True)
        model.solver.weight_decay(weight_decay)
        model.solver.update()

        model.train.loss.data.cast(np.float32, context)
        monitor.train.loss.add(i, model.train.loss.d.copy())

        model.train.pred.data.cast(np.float32, context)
        error = cl.categorical_error(model.train.pred.d, model.train.label.d)
        monitor.train.error.add(i, error)

        monitor.time.add(i)

    error = calc_error(model.val, data.val)
    monitor.val.error.add(i, error)
    nn.save_parameters(os.path.join(monitor.path, "params_%06d.h5" % (max_iter - 1)))
    return monitor


# 訓練の実行
for net in nets:
    print(f"[{net}]")
    model = create_model(net)
    train(model, max_iter=10000)

# 訓練結果の可視化
dfs = [read_monitor(get_monitor_path(net), melt="error", net=net) for net in nets]
df = pd.concat(dfs)
df.head()

# -
alt.Chart(df).mark_line(point=True).encode(
    x="step", y=alt.Y("error", scale={"type": "log"}), color="type"
).properties(width=200, height=150).facet("net")
