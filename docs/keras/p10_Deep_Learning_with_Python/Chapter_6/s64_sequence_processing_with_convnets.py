# # 6.4 Sequence processing with convnets
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/6.4-sequence-processing-with-convnets.ipynb)

import os

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import sequential

from ivory.utils.path import cache_dir
from ivory.utils.keras.history import plot_history

# ### Implementing a 1D convnet
max_features = 10000  # number of words to consider as features
max_len = 500  # cut texts after this number of words

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# -
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation="relu"))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation="relu"))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

# -
model.compile(optimizer=RMSprop(lr=1e-4), loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# -
plot_history(history, "acc") | plot_history(history, "loss")

# ### Combining CNNs and RNNs to process long sequences

base_dir = cache_dir("keras/ch6/data/weather/zip")
dfs = []
for name in os.listdir(os.path.join(base_dir)):
    df = pd.read_csv(os.path.join(base_dir, name), encoding="cp932")
    dfs.append(df)
df = pd.concat(dfs)
float_data = df.iloc[:, 1:].values
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(
    data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6
):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size,
)
val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size,
)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
    batch_size=batch_size,
)

# !This is how many steps to draw from `val_gen`
# !in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# !This is how many steps to draw from `test_gen`
# !in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

# -
model = Sequential()
model.add(
    layers.Conv1D(32, 5, activation="relu", input_shape=(None, float_data.shape[-1]))
)
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation="relu"))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation="relu"))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps,
)


# -
plot_history(history, "loss")

# -
# !This was previously set to 6 (one point per hour).
# !Now 3 (one point per 30 min).
step = 3
lookback = 720  # Unchanged
delay = 144  # Unchanged

train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
)
val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128
# -
model = Sequential()
model.add(
    layers.Conv1D(32, 5, activation="relu", input_shape=(None, float_data.shape[-1]))
)
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation="relu"))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()

# -
model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps,
)

# -
plot_history(history, "loss")
