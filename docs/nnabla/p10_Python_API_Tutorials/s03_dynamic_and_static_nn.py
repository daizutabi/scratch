# # Static vs Dynamic Neural Networks in NNabla
# # (https://nnabla.readthedocs.io/en/latest/python/tutorial/dynamic_and_static_nn.html)

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from nnabla.ext_utils import get_extension_context

from ivory.utils.repository import import_module

np.random.seed(0)
GPU = 0

# ## Dataset loading
tiny_digits = import_module("nnabla/tutorial/tiny_digits")
digits = tiny_digits.load_digits()
data = tiny_digits.data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)
# -
img, label = data.next()
print(img.shape, label.shape)


# ## Network definition
def cnn(x):
    """Unnecessarily Deep CNN.

    Args:
        x : Variable, shape (B, 1, 8, 8)

    Returns:
        y : Variable, shape (B, 10)
    """
    with nn.parameter_scope("cnn"):  # Parameter scope can be nested
        with nn.parameter_scope("conv1"):
            h = F.tanh(
                PF.batch_normalization(PF.convolution(x, 64, (3, 3), pad=(1, 1)))
            )
        for i in range(10):  # unnecessarily deep
            with nn.parameter_scope("conv{}".format(i + 2)):
                h = F.tanh(
                    PF.batch_normalization(PF.convolution(h, 128, (3, 3), pad=(1, 1)))
                )
        with nn.parameter_scope("conv_last"):
            h = F.tanh(
                PF.batch_normalization(PF.convolution(h, 512, (3, 3), pad=(1, 1)))
            )
            h = F.average_pooling(h, (2, 2))
        with nn.parameter_scope("fc"):
            h = F.tanh(PF.affine(h, 1024))
        with nn.parameter_scope("classifier"):
            y = PF.affine(h, 10)
    return y


# ### Static computation graph
# !setup cuda extension
ctx_cuda = get_extension_context('cudnn', device_id=GPU)
nn.set_default_context(ctx_cuda)

# !create variables for network input and label
x = nn.Variable(img.shape)
t = nn.Variable(label.shape)

# !create network
static_y = cnn(x)
static_y.persistent = True

# !define loss function for training
static_l = F.mean(F.softmax_cross_entropy(static_y, t))

# -
solver = S.Adam(alpha=1e-3)
solver.set_parameters(nn.get_parameters())

# -
loss = []  # type: ignore


def epoch_end_callback(epoch):
    global loss
    print("[{} {} {}]".format(epoch, np.mean(loss), itr))
    loss = []


data = tiny_digits.data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)
data.register_epoch_end_callback(epoch_end_callback)

# -
for epoch in range(30):
    itr = 0
    while data.epoch <= epoch:
        x.d, t.d = data.next()
        static_l.forward(clear_no_need_grad=True)
        solver.zero_grad()
        static_l.backward(clear_buffer=True)
        solver.update()
        loss.append(static_l.d.copy())
        itr += 1

# ### Dynamic computation graph
nn.clear_parameters()
solver = S.Adam(alpha=1e-3)
solver.set_parameters(nn.get_parameters())

data = tiny_digits.data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)
data.register_epoch_end_callback(epoch_end_callback)

# -
for epoch in range(30):
    itr = 0
    while data.epoch <= epoch:
        x.d, t.d = data.next()
        with nn.auto_forward():
            dynamic_y = cnn(x)
            dynamic_l = F.mean(F.softmax_cross_entropy(dynamic_y, t))

        # this can be done dynamically
        solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
        solver.zero_grad()
        dynamic_l.backward(clear_buffer=True)
        solver.update()
        loss.append(dynamic_l.d.copy())
        itr += 1
