# # Text classification with an RNN
# # (https://www.tensorflow.org/alpha/tutorials/sequences/text_classification_rnn)

import tensorflow_datasets as tfds

# ## Setup input pipeline
dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
