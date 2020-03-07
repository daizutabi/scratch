# # tf.function (https://www.tensorflow.org/alpha/tutorials/eager/tf_function)

import timeit

import tensorflow as tf


# !A function is like an op
@tf.function
def add(a, b):
    return a + b


add(tf.ones([2, 2]), tf.ones([2, 2]))  # [[2., 2.], [2., 2.]]


# !Functions have gradients
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)

# -
# !You can use functions inside functions
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)


dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))

# ## Polymorphism
# !Functions are polymorphic
@tf.function
def add_one(a):
    return a + a


print("add 1", add_one(1))
print("add 1.1", add_one(1.1))
print("add string tensor", add_one(tf.constant("a")))
c = add_one.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
c(a=tf.constant("a"))  # aa

# -
# !Functions can be faster than eager code, for graphs with many small ops
conv_layer = tf.keras.layers.Conv2D(100, 3)


@tf.function
def conv_fn(image):
    return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
# !warm up
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)


@tf.function
def lstm_fn(input, state):
    return lstm_cell(input, state)


input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# !warm up
lstm_cell(input, state)
lstm_fn(input, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))


# ## State in `tf.function`
# !Automatic control dependencies
a = tf.Variable(1.0)
b = tf.Variable(2.0)


@tf.function
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b


f(1.0, 2.0)  # 10.0


# ## Variables
@tf.function
def f2(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v


f2(1.0)  # Note: BROKEN, will throw exception
# Non-ambiguous code is ok though

v = tf.Variable(1.0)


@tf.function
def f3(x):
    return v.assign_add(x)


f3(1.0)  # 2.0
f3(2.0)  # 4.0

# -
# !You can also create variables inside a tf.function as long as we can prove
# !that those variables are created only the first time the function is executed.


class C:
    def __init__(self):
        self.v = None


obj = C()


@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)


g(1.0)  # 2.0
g(2.0)  # 4.0


# -
# !Variable initializers can depend on function arguments and on values of other
# !variables. We can figure out the right initialization order using the same
# !method we use to generate control dependencies.

state = []


@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]


fn(tf.constant(1.0))
fn(tf.constant(3.0))

# ## Control flow and autograph


# !Simple loop
@tf.function
def f4(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


f4(tf.random.uniform([10]))

# -
# !If you're curious you can inspect the code autograph generates.
# !It feels like reading assembly language, though.


def f5(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


print(tf.autograph.to_code(f5))
# -
@tf.function
def f6(x):
    for i in tf.range(10):
        tf.print(i)
        tf.Assert(i < 10, ["a"])
        x += x
    return x


f6(10)

# -
@tf.function
def f7(x):
    ta = tf.TensorArray(tf.float32, size=10)
    for i in tf.range(10):
        x += x
        ta = ta.write(i, x)
    return ta.stack()


f7(10.0)
