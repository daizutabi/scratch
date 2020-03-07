# # 6.3 Advanced usage of recurrent neural networks
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from ivory.utils.path import cache_dir
from ivory.utils.keras.history import plot_history

# ### A temperature forecasting problem

base_dir = cache_dir("keras/ch6/data/weather/zip")
for year in range(2009, 2017):
    for part in ["a", "b"]:
        url = f"https://www.bgc-jena.mpg.de/wetter/mpi_roof_{year}{part}.zip"
        path = os.path.join(base_dir, os.path.basename(url))
        res = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024):
                f.write(chunk)

dfs = []
for name in os.listdir(os.path.join(base_dir)):
    df = pd.read_csv(os.path.join(base_dir, name), encoding="cp932")
    dfs.append(df)
df = pd.concat(dfs)
float_data = df.iloc[:, 1:].values

# -
temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)

# -
plt.plot(range(1440), temp[:1440])

# ### Preparing the data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# -
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


# -
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


# ### A common sense, non-machine learning baseline
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()


# ### A basic machine learning approach
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation="relu"))
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

# ### A first recurrent baseline
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
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

# ### Using recurrent dropout to fight overfitting
model = Sequential()
model.add(
    layers.GRU(
        32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])
    )
)
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=50,  # 500 -> 50,
    epochs=5,  # 40 -> 5,
    validation_data=val_gen,
    validation_steps=val_steps,
)

# -
plot_history(history, "loss")

# ### Stacking recurrent layers
model = Sequential()
model.add(
    layers.GRU(
        32,
        dropout=0.1,
        recurrent_dropout=0.5,
        return_sequences=True,
        input_shape=(None, float_data.shape[-1]),
    )
)
model.add(layers.GRU(64, activation="relu", dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=50,  # 500 -> 50,
    epochs=5,  # 40 -> 5,
    validation_data=val_gen,
    validation_steps=val_steps,
)

# -
plot_history(history, "loss")


# ### Using bidirectional RNNs
def reverse_order_generator(
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
        yield samples[:, ::-1, :], targets


train_gen_reverse = reverse_order_generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size,
)
val_gen_reverse = reverse_order_generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size,
)


# -
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen_reverse,
    steps_per_epoch=50,  # 500 -> 50,
    epochs=5,  # 20 -> 5,
    validation_data=val_gen_reverse,
    validation_steps=val_steps,
)


# -
plot_history(history, "loss")

# -
# !Number of words to consider as features
max_features = 10000
# !Cut texts after this number of words (among top max_features most common words)
maxlen = 500

# !Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# !Reverse sequences
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

# !Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# -
plot_history(history, "loss")

# -
K.clear_session()

# -
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# -
model = Sequential()
model.add(
    layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1]))
)
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss="mae")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=50,  # 500 -> 50,
    epochs=5,  # 40 -> 5,
    validation_data=val_gen,
    validation_steps=val_steps,
)

# -
plot_history(history, "loss")
