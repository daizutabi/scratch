# # 3.6 Classifying newswires: a multi-class classification example
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/3.6-classifying-newswires.ipynb)

import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

from ivory.utils.keras.history import plot_history

# ### The Reuters dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000
)


# -
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# !Note that our indices were offset by 3
# !because 0, 1 and 2 are reserved indices for "padding", "start of sequence",
# !and "unknown".
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
decoded_newswire

# ### Preparing the data


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# !Our vectorized training data
x_train = vectorize_sequences(train_data)
# !Our vectorized test data
x_test = vectorize_sequences(test_data)


# -
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# ### Building our network
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

# -
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# ### Validating our approach
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# -
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)

# -
plot_history(history, "loss") | plot_history(history, "accuracy")

# -
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    partial_x_train,
    partial_y_train,
    epochs=8,
    batch_size=512,
    validation_data=(x_val, y_val),
)
results = model.evaluate(x_test, one_hot_test_labels)
results


# ### Generating predictions on new data
predictions = model.predict(x_test)
predictions[0].shape
# -
np.sum(predictions[0])
# -
np.argmax(predictions[0])

# ### A different way to handle the labels and the loss
y_train = np.array(train_labels)
y_test = np.array(test_labels)
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# ### On the importance of having sufficiently large intermediate layers
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val),
)

# -
plot_history(history, "loss") | plot_history(history, "accuracy")
