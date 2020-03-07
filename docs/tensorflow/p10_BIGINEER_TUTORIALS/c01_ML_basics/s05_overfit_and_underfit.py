# # Explore overfitting and underfitting
# # (https://www.tensorflow.org/alpha/tutorials/keras/overfit_and_underfit)

import altair as alt
import numpy as np
import pandas as pd
from tensorflow import keras

# ## Download the IMDB dataset

# !Multi-hot-encoding
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(
    num_words=NUM_WORDS
)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# -
df = pd.DataFrame({"label": train_data[0]}).reset_index()
alt.Chart(df[:100]).mark_rule().encode(x="index", y="label").properties(height=100)


# ## Demonstrate overfitting

# ### Create a baseline model
baseline_model = keras.Sequential(
    [
        # `input_shape` is only required here so that `.summary` works.
        keras.layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
baseline_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
baseline_model.summary()

# -
baseline_history = baseline_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)

# ### Create a smaller model
smaller_model = keras.Sequential(
    [
        keras.layers.Dense(4, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

smaller_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
smaller_model.summary()

# -
smaller_history = smaller_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)

# ### Create a bigger model
bigger_model = keras.models.Sequential(
    [
        keras.layers.Dense(512, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

bigger_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

bigger_model.summary()

# -
bigger_history = bigger_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)

# ## Plot the training and validation loss


def report(model, history, key):
    dfs = []
    for key_ in [key, "val_" + key]:
        df = pd.DataFrame({"epoch": history.epoch, key: history.history[key_]})
        if key_.startswith("val_"):
            df["type"] = "validation"
        else:
            df["type"] = "train"
        dfs.append(df)
    df = pd.concat(dfs)
    df["model"] = model
    return df


def plot(model_history, key="binary_crossentropy"):
    df = pd.concat([report(model, history, key) for model, history in model_history])
    chart = alt.Chart(df).encode(x="epoch", y=key, color="model")
    return chart.mark_line().encode(detail="type") + chart.mark_point().encode(
        shape="type"
    )


plot(
    [
        ("baseline", baseline_history),
        ("smaller", smaller_history),
        ("bigger", bigger_history),
    ]
)


# ## Strategies to prevent overfitting

# ### Add weight regularization
l2_model = keras.models.Sequential(
    [
        keras.layers.Dense(
            16,
            kernel_regularizer=keras.regularizers.l2(0.001),
            activation="relu",
            input_shape=(NUM_WORDS,),
        ),
        keras.layers.Dense(
            16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"
        ),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

l2_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

l2_model_history = l2_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)
# -
plot([("baseline", baseline_history), ("l2_model", l2_model_history)])


# ### Add dropout
dpt_model = keras.models.Sequential(
    [
        keras.layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

dpt_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

dpt_model_history = dpt_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2,
)
# -

plot([("baseline", baseline_history), ("dpt_model", dpt_model_history)])
