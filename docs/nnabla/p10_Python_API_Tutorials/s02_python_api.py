# # NNabla Python API Demonstration Tutorial
# # (https://nnabla.readthedocs.io/en/latest/python/tutorial/python_api.html)

import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np

from ivory.utils.path import cache_file

# ## NdArray

a = nn.NdArray((2, 3, 4))
print(a.data)

# -
print("[Substituting random values]")
a.data = np.random.randn(*a.shape)
print(a.data)
print("[Slicing]")
a.data[0, :, ::2] = 0
print(a.data)
# -
a.fill(1)  # Filling all values with one.
print(a.data)
# -
b = nn.NdArray.from_numpy_array(np.ones(a.shape))
print(b.data)
# ## Variable
x = nn.Variable([2, 3, 4], need_grad=True)
print("x.data:", x.data)
print("x.grad:", x.grad)
# -
x.shape
# -
print("x.data")
print(x.d)
x.d = 1.2345  # To avoid NaN
assert np.all(x.d == x.data.data), "d: {} != {}".format(x.d, x.data.data)
print("x.grad")
print(x.g)
x.g = 1.2345  # To avoid NaN
assert np.all(x.g == x.grad.data), "g: {} != {}".format(x.g, x.grad.data)

# !Zeroing grad values
x.grad.zero()
print("x.grad (after `.zero()`)")
print(x.g)
# -
x2 = nn.Variable.from_numpy_array(np.ones((3,)), need_grad=True)
print(x2)
print(x2.d)
x3 = nn.Variable.from_numpy_array(np.ones((3,)), np.zeros((3,)), need_grad=True)
print(x3)
print(x3.d)
print(x3.g)
# -
print(x.parent)

# ## Function
sigmoid_output = F.sigmoid(x)
sum_output = F.reduce_sum(sigmoid_output)
print(sigmoid_output)
print(sum_output)
# -
print("sigmoid_output.parent.name:", sigmoid_output.parent.name)
print("x:", x)
print("sigmoid_output.parent.inputs refers to x:", sigmoid_output.parent.inputs)
# -
print("sum_output.parent.name:", sum_output.parent.name)
print("sigmoid_output:", sigmoid_output)
print("sum_output.parent.inputs refers to sigmoid_output:", sum_output.parent.inputs)
# -
sum_output.forward()
print("CG output:", sum_output.d)
print("Reference:", np.sum(1.0 / (1.0 + np.exp(-x.d))))
# -
x.grad.zero()
sum_output.backward()
print("d sum_o / d sigmoid_o:")
print(sigmoid_output.g)
print("d sum_o / d x:")
print(x.g)
x.d
# -
x = nn.Variable([5, 2])  # Input
w = nn.Variable([2, 3], need_grad=True)  # Weights
b = nn.Variable([3], need_grad=True)  # Biases
affine_out = F.affine(x, w, b)  # Create a graph including only affine
affine_out
# -
# !Set random input and parameters
x.d = np.random.randn(*x.shape)
w.d = np.random.randn(*w.shape)
b.d = np.random.randn(*b.shape)
# !Initialize grad
x.grad.zero()  # Just for showing gradients are not computed when need_grad=False.
w.grad.zero()
b.grad.zero()
# !Forward and backward
affine_out.forward()
affine_out.backward()
# -
print("F.affine")
print(affine_out.d)
print("Reference")
print(np.dot(x.d, w.d) + b.d)
print("dw")
print(w.g)
print("db")
print(b.g)
# -
print(x.g)
# ## Parametric Function
with nn.parameter_scope("affine1"):
    c1 = PF.affine(x, 3)
# -
nn.get_parameters()
# -
c1 = PF.affine(x, 3, name="affine1")
nn.get_parameters()
# -
c1.shape
# -
with nn.parameter_scope("foo"):
    h = PF.affine(x, 3)
    with nn.parameter_scope("bar"):
        h = PF.affine(h, 4)

with nn.parameter_scope("foo"):
    params = nn.get_parameters()
params
# -
with nn.parameter_scope("foo"):
    nn.clear_parameters()
nn.get_parameters()
# ## MLP Example For Explanation
nn.clear_parameters()
batchsize = 16
x = nn.Variable([batchsize, 2])
with nn.parameter_scope("fc1"):
    h = F.tanh(PF.affine(x, 512))
with nn.parameter_scope("fc2"):
    y = PF.affine(h, 1)
print("Shapes:", h.shape, y.shape)
# -
nn.get_parameters()
# -
x.d = np.random.randn(*x.shape)  # Set random input
y.forward()
y.d
# -
# !Variable for label
label = nn.Variable([batchsize, 1])
# !Set loss
loss = F.reduce_mean(F.squared_error(y, label))

# !Execute forward pass.
label.d = np.random.randn(*label.shape)  # Randomly generate labels
loss.forward()
print(loss.d)
# -
# !Collect all parameter variables and init grad.
for name, param in nn.get_parameters().items():
    param.grad.zero()
# Gradients are accumulated to grad of params.
loss.backward()

# ## Imperative Mode
for name, param in nn.get_parameters().items():
    param.data -= param.grad * 0.001  # 0.001 as learning rate
# -
# !A simple example of imperative mode.
xi = nn.NdArray.from_numpy_array(np.arange(4).reshape(2, 2))
yi = F.relu(xi - 1)
xi.data
# -
yi.data
# -
id(xi)
# -
xi = xi + 1
id(xi)
# -
xi -= 1
id(xi)
# -
# !The following doesn't perform substitution but assigns a new NdArray object to `xi`.
# !xi = xi + 1

# !The following copies the result of `xi + 1` to `xi`.
xi.copy_from(xi + 1)
assert np.all(xi.data == (np.arange(4).reshape(2, 2) + 1))

# Inplace operations like `+=`, `*=` can also be used (more efficient).
xi += 1
assert np.all(xi.data == (np.arange(4).reshape(2, 2) + 2))

# ## Solver
solver = S.Sgd(lr=0.00001)
solver.set_parameters(nn.get_parameters())
# -
# !Set random data
x.d = np.random.randn(*x.shape)
label.d = np.random.randn(*label.shape)

# !Forward
loss.forward()

# -
solver.zero_grad()
loss.backward()
solver.update()


# ## Toy Problem To Demonstrate Training
def vector2length(x):
    # x : [B, 2] where B is number of samples.
    return np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))


# Example
vector2length(np.array([[3, 4], [5, 12]]))

# -
# !Data for plotting contour on a grid data.
xs = np.linspace(-1, 1, 100)
ys = np.linspace(-1, 1, 100)
grid = np.meshgrid(xs, ys)
X = grid[0].flatten()
Y = grid[1].flatten()


def plot_true():
    """Plotting contour of true mapping from a grid data created above."""
    plt.contourf(
        xs, ys, vector2length(np.hstack([X[:, None], Y[:, None]])).reshape(100, 100)
    )
    plt.axis("equal")
    plt.colorbar()


plot_true()


# -
def length_mlp(x):
    h = x
    for i, hnum in enumerate([4, 8, 4, 2]):
        h = F.tanh(PF.affine(h, hnum, name="fc{}".format(i)))
    y = PF.affine(h, 1, name="fc")
    return y


# -
nn.clear_parameters()
batchsize = 100
x = nn.Variable([batchsize, 2])
y = length_mlp(x)
label = nn.Variable([batchsize, 1])
loss = F.reduce_mean(F.squared_error(y, label))


# -
def predict(inp):
    ret = []
    for i in range(0, inp.shape[0], x.shape[0]):
        xx = inp[i : i + x.shape[0]]
        # Imperative execution
        xi = nn.NdArray.from_numpy_array(xx)
        yi = length_mlp(xi)
        ret.append(yi.data.copy())
    return np.vstack(ret)


def plot_prediction():
    plt.contourf(xs, ys, predict(np.hstack([X[:, None], Y[:, None]])).reshape(100, 100))
    plt.colorbar()
    plt.axis("equal")


# -
solver = S.Adam(alpha=0.01)
solver.set_parameters(nn.get_parameters())


# -
def random_data_provider(n):
    x = np.random.uniform(-1, 1, size=(n, 2))
    y = vector2length(x)
    return x, y


# -
num_iter = 2000
for i in range(num_iter):
    # Sample data and set them to input variables of training.
    xx, ll = random_data_provider(batchsize)
    x.d = xx
    label.d = ll
    # Forward propagation given inputs.
    loss.forward(clear_no_need_grad=True)
    # Parameter gradients initialization and gradients computation by backprop.
    solver.zero_grad()
    loss.backward(clear_buffer=True)
    # Apply weight decay and update by Adam rule.
    solver.weight_decay(1e-6)
    solver.update()
    # Just print progress.
    if i % 100 == 0 or i == num_iter - 1:
        print("Loss@{:4d}: {}".format(i, loss.d))
# -
loss.forward(clear_buffer=True)
print("The prediction `y` is cleared because it's an intermediate variable.")
print(y.d.flatten()[:4])  # to save space show only 4 values
y.persistent = True
loss.forward(clear_buffer=True)
print("The prediction `y` is kept by the persistent flag.")
print(y.d.flatten()[:4])  # to save space show only 4 value
# -
plt.subplot(121)
plt.title("Ground truth")
plot_true()
plt.subplot(122)
plt.title("Prediction")
plot_prediction()
# -
path_param = cache_file('nnabla/tutorial/python_api/param-vector2length.h5')
nn.save_parameters(path_param)
# !Remove all once
nn.clear_parameters()
nn.get_parameters()
# -
# !Load again
nn.load_parameters(path_param)
print('\n'.join(map(str, nn.get_parameters().items())))
# -
with nn.parameter_scope('foo'):
    nn.load_parameters(path_param)
print('\n'.join(map(str, nn.get_parameters().items())))
