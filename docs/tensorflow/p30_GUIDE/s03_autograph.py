# # tf.function and AutoGraph in TensorFlow 2.0
# # (https://www.tensorflow.org/alpha/guide/autograph)

import tensorflow as tf
from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

# ## The `tf.function` decorator
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)
# -
simple_nn_layer


# -
def linear_layer(x):
    return 2 * x + 1


@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))


deep_net(tf.constant((1, 2, 3)))

# ## Use Python control flow
@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x


print("square_if_positive(2) = {}".format(square_if_positive(tf.constant(2))))
print("square_if_positive(-2) = {}".format(square_if_positive(tf.constant(-2))))
# -
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s


sum_even(tf.constant([10, 12, 15, 20]))
# -
print(
    tf.autograph.to_code(sum_even.python_function, experimental_optional_features=None)
)


# ## Keras and AutoGraph
class CustomModel(tf.keras.models.Model):
    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2


model = CustomModel()

model(tf.constant([-2, -4]))
# ## Side effects
v = tf.Variable(5)


@tf.function
def find_next_odd():
    v.assign(v + 1)
    if tf.equal(v % 2, 0):
        v.assign(v + 1)


find_next_odd()
v


# ## Example: training a simple model
# ### Download data
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds


train_dataset = mnist_dataset()


# ### Define the model
model = tf.keras.Sequential(
    (
        tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10),
    )
)
model.build()
optimizer = tf.keras.optimizers.Adam()


# ### Define the training loop
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss


@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if tf.equal(step % 10, 0):
            tf.print(
                "Step", step, ": loss", loss, "; accuracy", compute_accuracy.result()
            )
    return step, loss, accuracy


step, loss, accuracy = train(model, optimizer)
print("Final step", step, ": loss", loss, "; accuracy", compute_accuracy.result())


# ## Batching
def square_if_positive(x):  # type: ignore
    return [i ** 2 if i > 0 else i for i in x]


square_if_positive(range(-5, 5))

# -
@tf.function
def square_if_positive_naive(x):
    result = tf.TensorArray(tf.int32, size=x.shape[0])
    for i in tf.range(x.shape[0]):
        if x[i] > 0:
            result = result.write(i, x[i] ** 2)
        else:
            result = result.write(i, x[i])
    return result.stack()


square_if_positive_naive(tf.range(-5, 5))


# -
def square_if_positive_vectorized(x):
    return tf.where(x > 0, x ** 2, x)


square_if_positive_vectorized(tf.range(-5, 5))
