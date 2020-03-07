# # Text classification with movie reviews
# # (https://www.tensorflow.org/alpha/tutorials/keras/basic_text_classification)

import altair as alt
import pandas as pd
from tensorflow import keras

# ## Download the IMDB dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# ## Explore the data
print(f"Training entries: {len(train_data)}, labels: {len(test_labels)}")
print(train_data[0][:15])
len(train_data[0]), len(train_data[1])

# ## Convert the integers back to words
word_index = imdb.get_word_index()

# !The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


decode_review(train_data[0])[:60]

# ## Prepare the data
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=256
)
len(train_data[0]), len(train_data[1])


# ## Build the model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()

# -
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ## Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ## Train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1,
)

# ## Evaluate the model
model.evaluate(test_data, test_labels)

# ## Create a graph of accuracy and loss over time
history_dict = history.history
df = pd.DataFrame(history_dict)
df.index.name = "epochs"
df.reset_index(inplace=True)
df.head()

# -
chart = alt.Chart(df).encode(x="epochs")
c1 = chart.mark_point().encode(y="accuracy")
c2 = chart.mark_line().encode(y="val_accuracy")
c1 + c2
