# # Transfer Learning Using Pretrained ConvNets
# # (https://www.tensorflow.org/alpha/tutorials/images/transfer_learning)

import tensorflow_datasets as tfds

# ## Data preprocessing
# ### Data download

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    "cats_vs_dogs", split=list(splits), with_info=True, as_supervised=True
)

# -
print(raw_train)
print(raw_validation)
print(raw_test)
