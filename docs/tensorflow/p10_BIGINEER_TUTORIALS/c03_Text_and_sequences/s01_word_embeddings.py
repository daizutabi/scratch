# #!

# # Word embeddings
# # (https://www.tensorflow.org/alpha/tutorials/sequences/word_embeddings)

from tensorflow import keras
from tensorflow.keras import layers

from ivory.utils.keras.history import plot_history

# ## Using the Embedding layer

# !The Embedding layer takes at least two arguments: the number of possible words in the
# !vocabulary, here 1000 (1 + maximum word index), and the dimensionality of the
# !embeddings, here 32.
embedding_layer = layers.Embedding(1000, 32)

# ### Learning embeddings from scratch
vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocab_size
)
train_data[0][:10]

# ### Convert the integers back to words
# !A dictionary mapping words to an integer index
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


decode_review(train_data[0])[:50]

# -
maxlen = 500

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=maxlen
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=maxlen
)

# ### Create a simple model
embedding_dim = 16
model = keras.Sequential(
    [
        layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()
# ### Compile and train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_data, train_labels, epochs=30, batch_size=512, validation_split=0.2
)


plot_history(history, "accuracy")

# ### Retrieve the learned embeddings
e = model.layers[0]
weights = e.get_weights()[0]
weights.shape  # shape: (vocab_size, embedding_dim)
