# # 4.4 Overfitting and underfitting
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/4.4-overfitting-and-underfitting.ipynb)

import altair as alt
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, regularizers

from ivory.utils.keras.history import history_to_dataframe

# -

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results


# !Our vectorized training data
x_train = vectorize_sequences(train_data)
# !Our vectorized test data
x_test = vectorize_sequences(test_data)
# !Our vectorized labels
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


# ### Fighting overfitting
def build_model(hidden_layer):
    model = models.Sequential()
    model.add(layers.Dense(hidden_layer, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(hidden_layer, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


def fit(model):
    return model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_test, y_test),
        verbose=0,
    )


# ### #Reducing the network's size
original_model = build_model(16)
smaller_model = build_model(4)
bigger_model = build_model(512)
histories = [fit(model) for model in [original_model, smaller_model, bigger_model]]
history_dict = {
    name: histories[k] for k, name in enumerate(["original", "smaller", "bigger"])
}


# -
df = history_to_dataframe(history_dict, "loss")
df.tail(3)

# -
chart = (
    alt.Chart()
    .mark_point()
    .encode(x="epoch", y="loss", shape="name", color="name")
    .properties(width=250, height=200)
)
charts = [
    chart.transform_filter(alt.datum.type == type_).properties(title=type_)
    for type_ in ["validation", "train"]
]
alt.hconcat(*charts, data=df)

# ### Adding weight regularization
l2_model = models.Sequential()
l2_model.add(
    layers.Dense(
        16,
        kernel_regularizer=regularizers.l2(0.001),
        activation="relu",
        input_shape=(10000,),
    )
)
l2_model.add(
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation="relu")
)
l2_model.add(layers.Dense(1, activation="sigmoid"))
l2_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

# -
history_dict["l2"] = l2_model.fit(
    x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test)
)

# -
df = history_to_dataframe(history_dict, "loss")
df = df.query('name in ["original", "l2"] and type == "validation"')
alt.Chart(df).mark_point().encode(x="epoch", y="loss", color="name").properties(
    width=250, height=200
)


# ### Adding dropout
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation="relu"))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation="sigmoid"))
dpt_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])


# -
history_dict["dpt"] = dpt_model.fit(
    x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test)
)

# -
df = history_to_dataframe(history_dict, "loss")
df = df.query('name in ["original", "dpt"] and type == "validation"')
alt.Chart(df).mark_point().encode(x="epoch", y="loss", color="name").properties(
    width=250, height=200
)
