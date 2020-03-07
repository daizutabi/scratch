# # Automatic differentiation and gradient tape
# # (https://www.tensorflow.org/alpha/tutorials/eager/automatic_differentiation)

import tensorflow as tf

# ## Gradient tapes
x = tf.ones((2, 2))
x

# -
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
print(y)
print(z)

# !Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
dz_dx

# -
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
# !Use the tape to compute the derivative of z with respect to the
# !intermediate value y.
dz_dy = t.gradient(z, y)
dz_dy

# -
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:  # Allow multiple calls to the `gradient`
    t.watch(x)
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape
print(dz_dx)
print(dy_dx)


# ## Recording control flow
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output


def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)


x = tf.convert_to_tensor(2.0)
assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0
x

# ## Higher-order gradients
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0
print(x)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0


# Compare: Variable vs. Tensor
x = tf.constant(1.0)
print(x)

with tf.GradientTape() as t:
    t.watch(x)
    with tf.GradientTape() as t2:
        t2.watch(x)
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
