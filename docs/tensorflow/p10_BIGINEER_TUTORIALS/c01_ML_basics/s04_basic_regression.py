# # Predict fuel efficiency: regression
# # (https://www.tensorflow.org/alpha/tutorials/keras/basic_regression)

import altair as alt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ## The Auto MPG dataset
# ### Get the data
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
)
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)
dataset = raw_dataset.copy()
dataset.tail()

# ### Clean the data
dataset.isna().sum()
# -
dataset = dataset.dropna()
# -
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
dataset.tail()

# ### Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ### Inspect the data
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
)

# -
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# ### Split features from labels
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# ### Normalize the data
def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# ## The model
# ### Build the model


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(
                64, activation="relu", input_shape=[len(train_dataset.keys())]
            ),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


model = build_model()


# ### Inspect the model
model.summary()

# -
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# ### Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()],
)


# -
def history_dataframe(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    df_train = hist[["epoch", "loss", "mae", "mse"]].copy()
    df_train["type"] = "train"
    df_val = hist[["epoch", "val_loss", "val_mae", "val_mse"]].copy()
    df_val.rename(
        columns={f"val_{key}": key for key in ["loss", "mae", "mse"]}, inplace=True
    )
    df_val["type"] = "test"
    df = pd.concat([df_train, df_val])
    return df


def plot(df):
    chart1 = (
        alt.Chart(df)
        .mark_line(clip=True)
        .encode(x="epoch", y=alt.Y("mae", scale=alt.Scale(domain=(0, 5))), color="type")
    )
    chart2 = (
        alt.Chart(df)
        .mark_line(clip=True)
        .encode(
            x="epoch", y=alt.Y("mse", scale=alt.Scale(domain=(0, 20))), color="type"
        )
    )
    return chart1, chart2


# -
df = history_dataframe(history)
c1, c2 = plot(df)
c1 | c2

# -
model = build_model()

# !The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()],
)
# -
df = history_dataframe(history)
c1, c2 = plot(df)
c1 | c2

# -
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# ## Make predictions
test_predictions = model.predict(normed_test_data).flatten()
df = pd.DataFrame({"label": test_labels, "pred": test_predictions})
df.head()

# -
chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(x="label", y="pred")
    .properties(width=250, height=250)
)
line = pd.DataFrame([[0, 0], [50, 50]], columns=["label", "pred"])
chart + alt.Chart(line).mark_line().encode(x="label", y="pred")

# -
error = (test_predictions - test_labels).to_frame()
error.head()

# -
alt.Chart(error).mark_bar().encode(alt.X("MPG:Q", bin={"maxbins": 40}), y="count()")
