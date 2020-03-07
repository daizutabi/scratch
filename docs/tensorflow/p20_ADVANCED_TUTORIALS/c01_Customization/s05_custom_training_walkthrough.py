# # Custom training: walkthrough
# # (https://www.tensorflow.org/alpha/tutorials/eager/custom_training_walkthrough)

# ## TensorFlow programming
# This guide uses these high-level TensorFlow concepts:

# * Use TensorFlow's default eager execution development environment,
# * Import data with the Datasets API,
# * Build models and layers with TensorFlow's Keras API.

# This tutorial is structured like many TensorFlow programs:

# 1. Import and parse the data sets.
# 1. Select the type of model.
# 1. Train the model.
# 1. Evaluate the model's effectiveness.
# 1. Use the trained model to make predictions.

# ## Setup program
# ### Configure imports

import os

import altair as alt
import pandas as pd
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# ## Import and parse the training dataset
# ### Download the dataset
train_dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url
)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# ### Inspect the data
df = pd.read_csv(train_dataset_fp)
# !column order in CSV file
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df.columns = column_names
df.head()
# -
feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]

# ### Create a `tf.data.Dataset`
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
)
# -
features, labels = next(iter(train_dataset))
print(features)
# -
df = pd.DataFrame(features)
df["label"] = labels
alt.Chart(df).mark_circle().encode(
    x="petal_length", y="sepal_length", color="label:N"
).properties(width=200, height=150)


# -
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:5])

# ## Select the type of model
# ### Create a model using Keras
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            10, activation=tf.nn.relu, input_shape=(4,)
        ),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3),
    ]
)

# ### Using the model
predictions = model(features)
predictions[:5]
# -
tf.nn.softmax(predictions[:5])
# -
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# ## Train the model
# ### Define the loss and gradient function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))


# -
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# ### Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)

print(
    "Step: {}, Initial Loss: {}".format(
        optimizer.iterations.numpy(), loss_value.numpy()
    )
)

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(
    "Step: {},         Loss: {}".format(
        optimizer.iterations.numpy(), loss(model, features, labels).numpy()
    )
)
# ### Training loop
# With all the pieces in place, the model is ready for training! A training loop feeds
# the dataset examples into the model to help it make better predictions. The following
# code block sets up these training steps:

# 1. Iterate each epoch. An epoch is one pass through the dataset.
# 1. Within an epoch, iterate over each example in the training Dataset grabbing its
# features (x) and label (y).
# 1. Using the example's features, make a prediction and compare it with the label.
# Measure the inaccuracy of the prediction and use that to calculate the model's loss
# and gradients.
# 1. Use an optimizer to update the model's variables.
# 1. Keep track of some stats for visualization.
# 1. Repeat for each epoch.

# !Note: Rerunning this cell uses the same model variables keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(y, model(x))

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print(
            "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result()
            )
        )

# ### Visualize the loss function over time
train_loss_array = [x.numpy() for x in train_loss_results]
train_accuracy_array = [x.numpy() for x in train_accuracy_results]
df = pd.DataFrame({"loss": train_loss_array, "accuracy": train_accuracy_array})
df.index.name = "epoch"
df.reset_index(inplace=True)
chart = alt.Chart(df).mark_line().encode(x="epoch").properties(height=100)
chart.encode(y="loss") & chart.encode(y="accuracy")

# ## Evaluate the model's effectiveness
# ### Setup the test dataset
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name="species",
    num_epochs=1,
    shuffle=False,
)

test_dataset = test_dataset.map(pack_features_vector)

# ### Evaluate the model on the test dataset
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# ## Use the trained model to make predictions
predict_dataset = tf.convert_to_tensor(
    [[5.1, 3.3, 1.7, 0.5], [5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1]]
)

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))
