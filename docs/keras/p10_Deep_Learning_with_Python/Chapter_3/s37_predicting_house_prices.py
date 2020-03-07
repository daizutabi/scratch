# # 3.7 Predicting house prices: a regression example
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/3.7-predicting-house-prices.ipynb)

import altair as alt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

from ivory.utils.keras.history import history_to_dataframe

# ### The Boston Housing Price dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
train_data.shape, test_data.shape

# ### Preparing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# ### Building our network
def build_model():
    # Because we will need to instantiate the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# ### Validating our approach using K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print("processing fold #", i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples :]],
        axis=0,
    )
    partial_train_targets = np.concatenate(
        [
            train_targets[: i * num_val_samples],
            train_targets[(i + 1) * num_val_samples :],
        ],
        axis=0,
    )

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(
        partial_train_data,
        partial_train_targets,
        epochs=num_epochs,
        batch_size=1,
        verbose=0,
    )
    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
# -
all_scores

# -
np.mean(all_scores)

# -
# !Some memory clean-up
K.clear_session()

# -
num_epochs = 500
histories = []
for i in range(k):
    print("processing fold #", i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples :]],
        axis=0,
    )
    partial_train_targets = np.concatenate(
        [
            train_targets[: i * num_val_samples],
            train_targets[(i + 1) * num_val_samples :],
        ],
        axis=0,
    )

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=0,
    )
    histories.append(history)
# -
dfs = []
for i in range(k):
    df = history_to_dataframe(histories[i], "mae")
    dfs.append(df)
df = pd.concat(dfs)
df = df.groupby(["epoch", "type"]).mean().reset_index()

alt.Chart(df).mark_line().encode(x="epoch", y="mae", color="type")
