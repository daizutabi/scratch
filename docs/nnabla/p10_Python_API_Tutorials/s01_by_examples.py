# #!

# # NNabla by Examples
# # (https://nnabla.readthedocs.io/en/latest/python/tutorial/by_examples.html)

import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import tile_images

from ivory.utils.path import cache_file
from ivory.utils.repository import import_module

tiny_digits = import_module("nnabla/tutorial/tiny_digits")
np.random.seed(0)
imshow_opt = dict(cmap="gray", interpolation="nearest")

# ## Logistic Regression
# ### Preparing a Toy Dataset
digits = tiny_digits.load_digits(n_class=10)
tiny_digits.plot_stats(digits)

# -
data = tiny_digits.data_iterator_tiny_digits(digits, batch_size=64, shuffle=True)
# -
img, label = data.next()
plt.imshow(tile_images(img), **imshow_opt)
print("labels:\n", label.reshape(8, 8))
print("Label shape:", label.shape)
# ### Preparing the Computation Graph
# !Forward pass
x = nn.Variable(img.shape)  # Define an image variable
with nn.parameter_scope("affine1"):
    y = PF.affine(x, 10)  # Output is 10 class
# -
# !Building a loss graph
t = nn.Variable(label.shape)  # Define an target variable
# !Softmax Xentropy fits multi-class classification problems
loss = F.mean(F.softmax_cross_entropy(y, t))

# -
print("Printing shapes of variables")
print(x.shape)
print(y.shape)
print(t.shape)
print(loss.shape)  # empty tuple means scalar

# ### Executing a static graph
# !Set data
x.d = img
t.d = label
# !Execute a forward pass
loss.forward()
# Showing results
print("Prediction score of 0-th image:", y.d[0])
print("Loss:", loss.d)

# ### Backward propagation through the graph
print(nn.get_parameters())
# -
for param in nn.get_parameters().values():
    print(param)
    param.grad.zero()
# -
# !Compute backward
loss.backward()
# !Showing gradients.
for name, param in nn.get_parameters().items():
    print(name, param.shape, param.g.flat[:20])  # Showing first 20.
# ### Optimizing parameters (=Training)
# !Create a solver (gradient-based optimizer)
learning_rate = 1e-3
solver = S.Sgd(learning_rate)
solver.set_parameters(nn.get_parameters())  # Set parameter variables to be updated.

# !One step of training
x.d, t.d = data.next()
loss.forward()
solver.zero_grad()  # Initialize gradients of all parameters to zero.
loss.backward()
solver.weight_decay(1e-5)  # Applying weight decay as an regularization
solver.update()
print(loss.d)

# -
for i in range(1000):
    x.d, t.d = data.next()
    loss.forward()
    solver.zero_grad()  # Initialize gradients of all parameters to zero.
    loss.backward()
    solver.weight_decay(1e-5)  # Applying weight decay as an regularization
    solver.update()
    if i % 100 == 0:  # Print for each 10 iterations
        print(i, loss.d)
# ### Show prediction
x.d, t.d = data.next()
y.forward()  # You can execute a sub graph.
plt.imshow(tile_images(x.d), **imshow_opt)
print("prediction:")
print(y.d.argmax(axis=1).reshape(8, 8))


# ### Dynamic graph construction support
def logreg_forward(x):
    with nn.parameter_scope("affine1"):
        y = PF.affine(x, 10)
    return y


def logreg_loss(y, t):
    # Softmax Xentropy fits multi-class classification problems
    loss = F.mean(F.softmax_cross_entropy(y, t))
    return loss


# -
x = nn.Variable(img.shape)
t = nn.Variable(label.shape)
x.d, t.d = data.next()
with nn.auto_forward():  # Graph are executed
    y = logreg_forward(x)
    loss = logreg_loss(y, t)
print("Loss:", loss.d)
plt.imshow(tile_images(x.d), **imshow_opt)
print("prediction:")
print(y.d.argmax(axis=1).reshape(8, 8))


# ## Multi-Layer Perceptron (MLP)
nn.clear_parameters()  # Clear all parameters


# -
def mlp(x, hidden=[16, 32, 16]):
    hs = []
    with nn.parameter_scope("mlp"):  # Parameter scope can be nested
        h = x
        for hid, hsize in enumerate(hidden):
            with nn.parameter_scope("affine{}".format(hid + 1)):
                h = F.tanh(PF.affine(h, hsize))
                hs.append(h)
        with nn.parameter_scope("classifier"):
            y = PF.affine(h, 10)
    return y, hs


# -
# !Construct a MLP graph
y, hs = mlp(x)

# -
print("Printing shapes")
print("x:", x.shape)
for i, h in enumerate(hs):
    print("h{}:".format(i + 1), h.shape)
print("y:", y.shape)

# -
# !Training
loss = logreg_loss(y, t)  # Reuse logreg loss function.


# !Copied from the above logreg example.
def training(steps, learning_rate):
    solver = S.Sgd(learning_rate)
    solver.set_parameters(nn.get_parameters())  # Set parameter variables to be updated.
    for i in range(steps):
        x.d, t.d = data.next()
        loss.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss.backward()
        solver.weight_decay(1e-5)  # Applying weight decay as an regularization
        solver.update()
        if i % 100 == 0:  # Print for each 10 iterations
            print(i, loss.d)


# !Training
training(1000, 1e-2)

# -
# !Showing responses for each layer
num_plot = len(hs) + 2
gid = 1


def scale01(h):
    return (h - h.min()) / (h.max() - h.min())


def imshow(img, title):
    global gid
    plt.subplot(num_plot, 1, gid)
    gid += 1
    plt.title(title)
    plt.imshow(img, **imshow_opt)
    plt.axis("off")


plt.figure(figsize=(2, 5))
imshow(x.d[0, 0], "x")
for hid, h in enumerate(hs):
    imshow(scale01(h.d[0]).reshape(-1, 8), "h{}".format(hid + 1))
imshow(scale01(y.d[0]).reshape(2, 5), "y")

# ## Convolutional Neural Network with CUDA acceleration
nn.clear_parameters()


# -
def cnn(x):
    with nn.parameter_scope("cnn"):  # Parameter scope can be nested
        with nn.parameter_scope("conv1"):
            c1 = F.tanh(
                PF.batch_normalization(
                    PF.convolution(x, 4, (3, 3), pad=(1, 1), stride=(2, 2))
                )
            )
        with nn.parameter_scope("conv2"):
            c2 = F.tanh(
                PF.batch_normalization(PF.convolution(c1, 8, (3, 3), pad=(1, 1)))
            )
            c2 = F.average_pooling(c2, (2, 2))
        with nn.parameter_scope("fc3"):
            fc3 = F.tanh(PF.affine(c2, 32))
        with nn.parameter_scope("classifier"):
            y = PF.affine(fc3, 10)
    return y, [c1, c2, fc3]


# -
# !Run on CUDA
cuda_device_id = 0
ctx = get_extension_context("cudnn", device_id=cuda_device_id)
print("Context:", ctx)
nn.set_default_context(ctx)  # Set CUDA as a default context.
y, hs = cnn(x)

# -
training(1000, 1e-1)

# -
# !Showing responses for each layer
num_plot = len(hs) + 2
gid = 1
plt.figure(figsize=(2, 8))
imshow(x.d[0, 0], "x")
imshow(tile_images(hs[0].d[0][:, None]), "conv1")
imshow(tile_images(hs[1].d[0][:, None]), "conv2")
imshow(hs[2].d[0].reshape(-1, 8), "fc3")
imshow(scale01(y.d[0]).reshape(2, 5), "y")


# -
path_cnn_params = cache_file("nnabla/tutorial/tmp.params.cnn.h5")
nn.save_parameters(path_cnn_params)

# ## Recurrent Neural Network (Elman RNN)
nn.clear_parameters()


# -
def rnn(xs, h0, hidden=32):
    hs = []
    with nn.parameter_scope("rnn"):
        h = h0
        # Time step loop
        for x in xs:
            # Note: Parameter scopes are reused over time
            # which means parameters are shared over time.
            with nn.parameter_scope("x2h"):
                x2h = PF.affine(x, hidden, with_bias=False)
            with nn.parameter_scope("h2h"):
                h2h = PF.affine(h, hidden)
            h = F.tanh(x2h + h2h)
            hs.append(h)
        with nn.parameter_scope("classifier"):
            y = PF.affine(h, 10)
    return y, hs


# -
def split_grid4(x):
    x0 = x[..., :4, :4]
    x1 = x[..., :4, 4:]
    x2 = x[..., 4:, :4]
    x3 = x[..., 4:, 4:]
    return x0, x1, x2, x3


# -
hidden = 32
seq_img = split_grid4(img)
seq_x = [nn.Variable(subimg.shape) for subimg in seq_img]
h0 = nn.Variable((img.shape[0], hidden))  # Initial hidden state.
y, hs = rnn(seq_x, h0, hidden)
loss = logreg_loss(y, t)


# -
# !Copied from the above logreg example.
def training_rnn(steps, learning_rate):
    solver = S.Sgd(learning_rate)
    solver.set_parameters(nn.get_parameters())  # Set parameter variables to be updated.
    for i in range(steps):
        minibatch = data.next()
        img, t.d = minibatch
        seq_img = split_grid4(img)
        h0.d = 0  # Initialize as 0
        for x, subimg in zip(seq_x, seq_img):
            x.d = subimg
        loss.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss.backward()
        solver.weight_decay(1e-5)  # Applying weight decay as an regularization
        solver.update()
        if i % 100 == 0:  # Print for each 10 iterations
            print(i, loss.d)


training_rnn(1000, 1e-1)

# -
# !Showing responses for each layer
num_plot = len(hs) + 2
gid = 1
plt.figure(figsize=(2, 8))
imshow(x.d[0, 0], "x")
for hid, h in enumerate(hs):
    imshow(scale01(h.d[0]).reshape(-1, 8), "h{}".format(hid + 1))
imshow(scale01(y.d[0]).reshape(2, 5), "y")

# ## Siamese Network
nn.clear_parameters()
# !Loading CNN pretrained parameters.
_ = nn.load_parameters(path_cnn_params)


# -
def cnn_embed(x, test=False):
    # Note: Identical configuration with the CNN example above.
    # Parameters pretrained in the above CNN example are used.
    with nn.parameter_scope("cnn"):
        with nn.parameter_scope("conv1"):
            c1 = F.tanh(
                PF.batch_normalization(
                    PF.convolution(x, 4, (3, 3), pad=(1, 1), stride=(2, 2)),
                    batch_stat=not test,
                )
            )
        with nn.parameter_scope("conv2"):
            c2 = F.tanh(
                PF.batch_normalization(
                    PF.convolution(c1, 8, (3, 3), pad=(1, 1)), batch_stat=not test
                )
            )
            c2 = F.average_pooling(c2, (2, 2))
        with nn.parameter_scope("fc3"):
            fc3 = PF.affine(c2, 32)
    # Additional affine for map into 2D.
    with nn.parameter_scope("embed2d"):
        embed = PF.affine(c2, 2)
    return embed, [c1, c2, fc3]


def siamese_loss(e0, e1, t, margin=1.0, eps=1e-4):
    dist = F.sum(F.squared_error(e0, e1), axis=1)  # Squared distance
    # Contrastive loss
    sim_cost = t * dist
    dissim_cost = (1 - t) * (F.maximum_scalar(margin - (dist + eps) ** (0.5), 0) ** 2)
    return F.mean(sim_cost + dissim_cost)


# -
x0 = nn.Variable(img.shape)
x1 = nn.Variable(img.shape)
t = nn.Variable((img.shape[0],))  # Same class or not
e0, hs0 = cnn_embed(x0)
e1, hs1 = cnn_embed(x1)  # NOTE: parameters are shared
loss = siamese_loss(e0, e1, t)


# -
def training_siamese(steps):
    for i in range(steps):
        minibatchs = []
        for _ in range(2):
            minibatch = data.next()
            minibatchs.append((minibatch[0].copy(), minibatch[1].copy()))
        x0.d, label0 = minibatchs[0]
        x1.d, label1 = minibatchs[1]
        t.d = (label0 == label1).astype(np.int).flat
        loss.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss.backward()
        solver.weight_decay(1e-5)  # Applying weight decay as an regularization
        solver.update()
        if i % 100 == 0:  # Print for each 10 iterations
            print(i, loss.d)


learning_rate = 1e-2
solver = S.Sgd(learning_rate)
with nn.parameter_scope("embed2d"):
    # Only 2d embedding affine will be updated.
    solver.set_parameters(nn.get_parameters())
training_siamese(2000)
# !Decay learning rate
solver.set_learning_rate(solver.learning_rate() * 0.1)
training_siamese(2000)

# -
all_image = digits.images[:512, None]
all_label = digits.target[:512]

# -
x_all = nn.Variable(all_image.shape)
x_all.d = all_image

# -
with nn.auto_forward():
    embed, _ = cnn_embed(x_all, test=True)
# -
plt.figure(figsize=(16, 9))
for i in range(10):
    c = plt.cm.Set1(i / 10.0)
    plt.plot(
        embed.d[all_label == i, 0].flatten(),
        embed.d[all_label == i, 1].flatten(),
        ".",
        c=c,
    )
plt.legend(map(str, range(10)))
plt.grid()
