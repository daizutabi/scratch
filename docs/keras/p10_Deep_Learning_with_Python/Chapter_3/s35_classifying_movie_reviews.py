# # 3.5 Classifying movie reviews: a binary classification example
# # (https://nbviewer.jupyter.org/github/fchollet/
# # deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb)

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, losses, metrics, models, optimizers

from ivory.utils.keras.history import history_to_dataframe, plot_history

# ### The IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
max([max(sequence) for sequence in train_data])

# !word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# !We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# !We decode the review; note that our indices were offset by 3 because 0, 1 and 2 are
# !reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
decoded_review


# ### Preparing the data
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results


# -
# !Our vectorized training data
x_train = vectorize_sequences(train_data)
# !Our vectorized test data
x_test = vectorize_sequences(test_data)

# -
# !Our vectorized labels
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


# ### Building our network
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# -
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# -
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy],
)


# ### Validating our approach
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# -
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)


# -
df = history_to_dataframe(history, "loss")
df.head(3)

# -
plot_history(history, "loss")

# -
df = history_to_dataframe(history, "binary_accuracy")
df.head(3)

# -
plot_history(history, "binary_accuracy")

# -
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# ### Using a trained network to generate predictions on new data
model.predict(x_test)

# ### #Further experiments


def build_model(
    hidden_layer=2, hidden_unit=16, loss="binary_crossentropy", activation="relu"
):
    model = models.Sequential()
    model.add(layers.Dense(hidden_unit, activation=activation, input_shape=(10000,)))
    for k in range(hidden_layer - 1):
        model.add(layers.Dense(hidden_unit, activation=activation))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss=loss, metrics=["accuracy"])
    return model


def fit(model, epochs=20, batch_size=512, verbose=0):
    return model.fit(
        partial_x_train,
        partial_y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=verbose,
    )


# #Experiment 3.5.1 We were using 2 hidden layers. Try to use 1 or 3 hidden layers and
# #Experiment see how it affects validation and test accuracy.

histories = [fit(build_model(hidden_layer=k), epochs=20) for k in range(1, 4)]

charts = [plot_history(history, "accuracy") for history in histories]
charts[0] | charts[1] | charts[2]

# ###Experiment Try to use layers with more hidden units or less hidden units: 32 units,
# ###Experiment 64 units...
...

# ###Experiment Try to use the mse loss function instead of binary_crossentropy.
...

# ###Experiment Try to use the tanh activation (an activation that was popular in the
# ###Experiment early days of neural networks) instead of relu.
...
