# # Transformer model for language understanding
# # (https://www.tensorflow.org/alpha/tutorials/text/transformer)

import tensorflow_datasets as tfds

# ## Setup input pipeline
examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)
train_examples, val_examples = examples["train"], examples["validation"]
