# # 6.2 Understanding recurrent neural networks
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/6.2-understanding-recurrent-neural-networks.ipynb)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential

from ivory.utils.keras.history import plot_history

# ### A first recurrent layer in Keras
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()
# -
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()
# -
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()
# -
max_features = 1000  # number of words to consider as features: 10000 -> 1000
maxlen = 500  # cut texts after this number of words
batch_size = 128

print("Loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), "train sequences")
print(len(input_test), "test sequences")

print("Pad sequences (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)

# ### A concrete LSTM example in Keras
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    input_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2
)

# -
plot_history(history, "acc") | plot_history(history, "loss")
