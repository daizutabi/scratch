# # Custom training: basics
# # (https://www.tensorflow.org/alpha/tutorials/eager/custom_training)

import altair as alt
import pandas as pd
import tensorflow as tf

# ## Variables
# !Using python state
x = tf.zeros([10, 10])
print(id(x))
x += 2
print(id(x))
print(x)
# -
v = tf.Variable(1.0)
print(id(v))
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0
print(id(v))


# ## Example: Fitting a linear model
# ### Define the model
class Model:
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
assert model(3.0).numpy() == 15.0


# ### Define a loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# ### Obtain training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

df = pd.DataFrame({"x": inputs, "y": outputs})
df.head()
# -
alt.Chart(df).mark_point().encode(x="x", y="y").properties(width=200, height=150)
# -
print("Current loss: ")
print(loss(model(inputs), outputs).numpy())


# ## Define a training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


# -
model = Model()

# !Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print(
        "Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss)
    )
# !Let's plot it all
df = pd.DataFrame({"epoch": epochs, "W": Ws, "b": bs})

chart = alt.Chart(df).mark_line().encode(x='epoch')
chart.encode(y="W") + chart.encode(y='b').properties(width=200, height=150)
