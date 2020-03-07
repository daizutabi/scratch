# # 6.1 Using word embeddings
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/6.1-using-word-embeddings.ipynb)

import os
import tarfile
import zipfile

import numpy as np
import requests
from tensorflow.keras import preprocessing
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential

from ivory.utils.path import cache_dir
from ivory.utils.keras.history import plot_history

# ### Learning word embeddings with the `Embedding` layer

# !The Embedding layer takes at least two arguments:
# !the number of possible tokens, here 1000 (1 + maximum word index),
# !and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)

# -
# !Number of words to consider as features
max_features = 10000
# !Cut texts after this number of words (among top max_features most common words)
maxlen = 20

# !Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# !This turns our lists of integers into a 2D integer tensor of
# !shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# -
model = Sequential()
# !We specify the maximum input length to our Embedding layer
# !so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))
# !After the Embedding layer, our activations have shape `(samples, maxlen, 8)`.

# !We flatten the 3D tensor of embeddings
# !into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# !We add the classifier on top
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ### Putting it all together: from raw text to word embeddings

# #### ##Download the IMDB data as raw text

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
base_dir = cache_dir("keras/ch6/data")
path = os.path.join(base_dir, os.path.basename(url))
res = requests.get(url, stream=True)
if res.status_code == 200:
    with open(path, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            f.write(chunk)

with tarfile.open(path, "r:gz") as tarf:
    tarf.extractall(path=base_dir)

# -
imdb_dir = os.path.join(base_dir, "aclImdb")
train_dir = os.path.join(imdb_dir, "train")

labels = []
texts = []

for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == ".txt":
            with open(os.path.join(dir_name, fname), encoding="utf-8") as file:
                texts.append(file.read())
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)

# #### Tokenize the data

maxlen = 100  # We will cut reviews after 100 words
training_samples = 200  # We will be training on 200 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)  # type: ignore

# !Split the data into a training set and a validation set
# !But first, shuffle the data, since we started from data
# !where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]


# #### Download the GloVe word embeddings

url = "http://nlp.stanford.edu/data/glove.6B.zip"
base_dir = cache_dir("keras/ch6/data")
path = os.path.join(base_dir, os.path.basename(url))
res = requests.get(url, stream=True)
if res.status_code == 200:
    with open(path, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            f.write(chunk)

with zipfile.ZipFile(path) as zipf:
    zipf.extract("glove.6B.100d.txt", base_dir)

# #### Pre-process the embeddings
embeddings_index = {}
with open(os.path.join(base_dir, "glove.6B.100d.txt"), encoding="utf-8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

# -
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# #### Define a model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# #### Load the GloVe embeddings in the model
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# #### Train and evaluate
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val)
)
model.save_weights(os.path.join(base_dir, "pre_trained_glove_model.h5"))

# -
plot_history(history, "acc") | plot_history(history, "loss")

# -
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# -
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val)
)

# -
plot_history(history, "acc") | plot_history(history, "loss")

# -
test_dir = os.path.join(imdb_dir, "test")

labels = []
texts = []

for label_type in ["neg", "pos"]:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == ".txt":
            with open(os.path.join(dir_name, fname), encoding="utf-8") as file:
                texts.append(file.read())
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# -
model.load_weights(os.path.join(base_dir, "pre_trained_glove_model.h5"))
model.evaluate(x_test, y_test)
