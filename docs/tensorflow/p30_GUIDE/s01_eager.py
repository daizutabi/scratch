# #!

# # Eager Execution (https://www.tensorflow.org/alpha/guide/eager)
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ivory.utils.path import cache_dir

# ## Setup and basic usage
tf.executing_eagerly()
# -
x = [[2.0]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
# -
a = tf.constant([[1, 2], [3, 4]])
print(a)
# -
# !Broadcasting support
b = tf.add(a, 1)
print(b)
# -
# !Operator overloading is supported
print(a * b)
# -
# !Use NumPy values
c = np.multiply(a, b)
print(c)
# -
# !Obtain numpy value from a tensor:
print(a.numpy())


# ## Dynamic control flow
def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy() + 1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print("FizzBuzz")
        elif int(num % 3) == 0:
            print("Fizz")
        elif int(num % 5) == 0:
            print("Buzz")
        else:
            print(num.numpy())
        counter += 1


fizzbuzz(15)


# ## Build a model
class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units
        self.dynamic = True

    def build(self, input_shape):
        # The build method gets called the first time your layer is used.
        # Creating variables on build() allows you to make their shape depend
        # on the input shape and hence removes the need for the user to specify
        # full shapes. It is possible to create variables during __init__() if
        # you already know their full shapes.
        self.kernel = self.add_variable("kernel", [input_shape[-1], self.output_units])

    def call(self, input):
        # Override call() instead of __call__ so we can perform some bookkeeping.
        return tf.matmul(input, self.kernel)


# -
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
        tf.keras.layers.Dense(10),
    ]
)


# -
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, input):
        """Run the model."""
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)  # reuse variables from dense2 layer
        return result


model = MNISTModel()


# ## Eager training
# ### Computing gradients
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
# ### Train a model
# !Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (
        tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
        tf.cast(mnist_labels, tf.int64),
    )
)
dataset = dataset.shuffle(1000).batch(32)

# -
# !Build the model
mnist_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, [3, 3], activation="relu", input_shape=(None, None, 1)
        ),
        tf.keras.layers.Conv2D(16, [3, 3], activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
    ]
)
# -
for images, labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())
# -
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
# -
for (batch, (images, labels)) in enumerate(dataset.take(400)):
    if batch % 10 == 0:
        print(".", end="")
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
# -
plt.plot(loss_history)
plt.xlabel("Batch #")
plt.ylabel("Loss [entropy]")


# ### Variables and optimizers
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5.0, name="weight")
        self.B = tf.Variable(10.0, name="bias")

    def call(self, inputs):
        return inputs * self.W + self.B


# !A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise


# !The loss function to be optimized
def loss(model, inputs, targets):  # type: ignore
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


def grad(model, inputs, targets):  # type: ignore
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])


# !Define:
# !1. A model.
# !2. Derivatives of a loss function with respect to model parameters.
# !3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# !Training loop
for i in range(300):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 20 == 0:
        print(
            "Loss at step {:03d}: {:.3f}".format(
                i, loss(model, training_inputs, training_outputs)
            )
        )

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

# ## Use objects for state during eager execution
# ### Variables are objects
if tf.test.is_gpu_available():
    with tf.device("gpu:0"):
        v = tf.Variable(tf.random.normal([1000, 1000]))
        v = None  # v no longer takes up GPU memory

# ### Object-based saving
x = tf.Variable(10.0)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.0)  # type: ignore
checkpoint_path = cache_dir("tensorflow/eager")
checkpoint.save(os.path.join(checkpoint_path, "ckpt"))

# -
x.assign(11.0)  # type: ignore # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0

# -
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, [3, 3], activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
    ]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint_dir = cache_dir("tensorflow/eager/model")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer, model=model)

root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
# ### Object-oriented metrics
m = tf.keras.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5


# ## Advanced automatic differentiation topics
# ### Dynamic models
def line_search_step(fn, init_x, rate=1.0):
    with tf.GradientTape() as tape:
        # Variables are automatically recorded, but manually watch a tensor
        tape.watch(init_x)
        value = fn(init_x)
    grad = tape.gradient(value, init_x)
    grad_norm = tf.reduce_sum(grad * grad)
    init_value = value
    while value > init_value - rate * grad_norm:
        x = init_x - rate * grad
        value = fn(x)
        rate /= 2.0
    return x, value


# ### Custom gradients
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
    y = tf.identity(x)

    def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]

    return y, grad_fn


# -
def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))


def grad_log1pexp(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp(x)
    return tape.gradient(value, x)


# -
# !The gradient computation works fine at x = 0.
grad_log1pexp(tf.constant(0.0)).numpy()
# -
# !However, x = 100 fails because of numerical instability.
grad_log1pexp(tf.constant(100.0)).numpy()


# -
@tf.custom_gradient
def log1pexp_2(x):
    e = tf.exp(x)

    def grad(dy):
        return dy * (1 - 1 / (1 + e))

    return tf.math.log(1 + e), grad


def grad_log1pexp_2(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp_2(x)
    return tape.gradient(value, x)


# -
# !As before, the gradient computation works fine at x = 0.
grad_log1pexp_2(tf.constant(0.0)).numpy()

# -
# !And the gradient computation also works at x = 100.
grad_log1pexp_2(tf.constant(100.0)).numpy()


# ## Performance
def measure(x, steps):
    # TensorFlow initializes a GPU the first time it's used, exclude from timing.
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)
    # tf.matmul can return before completing the matrix multiplication
    # (e.g., can return after enqueing the operation on a CUDA stream).
    # The x.numpy() call below will ensure that all enqueued operations
    # have completed (and will also copy the result to host memory,
    # so we're including a little more than just the matmul operation
    # time).
    _ = x.numpy()
    end = time.time()
    return end - start


shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# !Run on CPU:
with tf.device("/cpu:0"):
    print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# !Run on GPU, if available:
if tf.test.is_gpu_available():
    with tf.device("/gpu:0"):
        print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
else:
    print("GPU: not found")

# -
if tf.test.is_gpu_available():
    x = tf.random.normal([10, 10])

    x_gpu0 = x.gpu()  # type: ignore
    x_cpu = x.cpu()  # type: ignore

    _ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
    _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0 0
